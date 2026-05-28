pub mod runtime_mod {
    use crate::backend::{Backend, CpuBackend, CudaBackend, WgpuBackend};
    use crate::graph::{get_indexed_inputs, Edge, GraphTensor, Node, Op};
    use crate::memory::registry::BufferRegistry;
    use crate::scheduler::{ScheduledOp, Scheduler, SimpleScheduler};
    use crate::tensor::{Shape, Tensor};
    use crate::{Device, Error};
    use std::sync::Arc;

    /// Executes the computation graph to produce the final value of the target tensor.
    /// Manages device fallback, operation scheduling, prefetching, memory eviction, and backend dispatch.
    pub fn execute(target: &GraphTensor, device: Device) -> Result<Tensor, Error> {
        execute_impl(target, device, false)
    }

    /// Executes the computation graph without the final GPU→CPU readback.
    /// Returns `()` after all compute kernels have been submitted.
    /// The result remains in GPU memory; call [`execute`] with the same graph to read it back.
    pub fn execute_no_readback(target: &GraphTensor, device: Device) -> Result<(), Error> {
        execute_impl(target, device, true)?;
        Ok(())
    }

    fn execute_impl(
        target: &GraphTensor,
        device: Device,
        skip_readback: bool,
    ) -> Result<Tensor, Error> {
        use crate::memory::planner::StaticMemoryPlanner;
        use crate::tensor::TensorId;
        use std::collections::HashMap;

        if let Some(err_msg) = target.graph.get_error() {
            return Err(Error::ExecutionError(err_msg));
        }
        // 1. Resolve device and backend
        let (_backend, target_device): (Box<dyn Backend>, Device) = match device {
            Device::Wgpu => {
                let backend = WgpuBackend::get_or_init()?;
                (Box::new(backend.clone()), Device::Wgpu)
            }
            Device::Cpu => (Box::new(CpuBackend::new()), Device::Cpu),
            Device::Auto => match WgpuBackend::get_or_init() {
                Ok(backend) => (Box::new(backend.clone()), Device::Wgpu),
                Err(err) => {
                    tracing::warn!(
                        "Failed to initialize WGPU backend, falling back to CPU: {:?}",
                        err
                    );
                    (Box::new(CpuBackend::new()), Device::Cpu)
                }
            },
            Device::Cuda => (Box::new(CudaBackend::new()), Device::Cuda),
        };

        // 2. Generate schedule
        let schedule = if target
            .graph
            .inner
            .read()
            .expect("Graph lock poisoned in execute_impl")
            .enable_fusion
        {
            crate::scheduler::fusion::FusionPass::run(target)?
        } else {
            let scheduler = SimpleScheduler::new();
            let plan = scheduler.schedule(target, target_device)?;
            plan.steps
                .into_iter()
                .map(|step| ScheduledOp::Plain(step.node_id, step.op))
                .collect()
        };

        // 3. Generate Prefetch/Eviction Plan
        let prefetch_plan =
            crate::memory::prefetch::PrefetchScheduler::plan(&schedule, &target.graph);

        // 4a. Compute liveness
        let liveness = {
            let inner = target.graph.inner.read().unwrap();
            let dag = &inner.dag;
            crate::memory::liveness::LivenessMap::analyze(&schedule, dag)
        };

        // 4b. Static Memory Planning
        // Build reshape alias map: Reshape(out) -> canonical input TensorId.
        // This lets the planner treat Reshape as a zero-copy alias.
        let reshape_aliases: HashMap<TensorId, TensorId> = {
            let inner = target.graph.inner.read().unwrap();
            let dag = &inner.dag;
            let mut map = HashMap::new();
            for op in &schedule {
                if let crate::scheduler::ScheduledOp::Plain(
                    node_id,
                    crate::graph::Op::Reshape { .. },
                ) = op
                {
                    let out_tid = dag[*node_id].tensor_id;
                    let inputs =
                        crate::graph::get_indexed_inputs(dag, *node_id).unwrap_or_default();
                    if let Some(&inp) = inputs.first() {
                        let in_tid = dag[inp].tensor_id;
                        map.insert(out_tid, in_tid);
                    }
                }
            }
            map
        };

        let (gpu_plan, cpu_plan) = {
            let inner = target.graph.inner.read().unwrap();
            let dag = &inner.dag;
            let target_tid = dag[target.id].tensor_id;

            let gpu_p = if target_device == Device::Wgpu {
                StaticMemoryPlanner::plan(
                    &schedule,
                    dag,
                    &liveness,
                    target_tid,
                    Device::Wgpu,
                    target_device,
                    &target.graph,
                    256,
                )
            } else {
                crate::memory::planner::StaticMemoryPlan {
                    allocations: HashMap::new(),
                    total_size: 0,
                }
            };

            let cpu_p = if target_device == Device::Cpu {
                StaticMemoryPlanner::plan(
                    &schedule,
                    dag,
                    &liveness,
                    target_tid,
                    Device::Cpu,
                    target_device,
                    &target.graph,
                    4, // 4-byte (f32) alignment for CPU
                )
            } else {
                crate::memory::planner::StaticMemoryPlan {
                    allocations: HashMap::new(),
                    total_size: 0,
                }
            };

            (gpu_p, cpu_p)
        };

        // 4c. Arena pre-allocation (computed for diagnostics and future custom kernel use).
        // NOTE: GPU arena dispatch is disabled: WebGPU does not permit the same buffer object
        //       to be bound as STORAGE_READ and STORAGE_READ_WRITE simultaneously, even at
        //       different byte offsets. The arena pointer and plan are available for profiling.
        // NOTE: CPU arena dispatch is disabled pending integration with ensure_cpu path.
        let _gpu_arena_size = gpu_plan.total_size;
        let _cpu_arena_size = cpu_plan.total_size;

        // 5. Initialize BufferRegistry (classic dynamic mode — arenas passed as None)
        let limit = target.graph.gpu_byte_limit();
        let registry = BufferRegistry::new(limit);

        // 6. Register all input tensors in registry
        for input_tensor in target.graph.input_tensors() {
            registry.register_cpu(
                input_tensor.id(),
                input_tensor.data_raw().to_f32(),
                input_tensor.shape().clone(),
                input_tensor.dtype(),
            );
        }

        let cpu_backend = CpuBackend::new();
        let _ = &reshape_aliases; // suppress unused warning; used by arena dispatch functions

        let (final_data, final_shape) = {
            let inner_guard = target.graph.inner.read().unwrap();
            let dag = &inner_guard.dag;
            let target_tid = dag[target.id].tensor_id;

            // 7. Parallel Execution Graph Dispatch via Rayon
            use std::collections::HashMap;
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::mpsc;
            use std::sync::{Arc, Mutex};

            // Helper to get inputs/output of a ScheduledOp
            let get_op_io = |op: &ScheduledOp| -> (Vec<TensorId>, TensorId) {
                match op {
                    ScheduledOp::Plain(node_id, _) => {
                        let inputs =
                            crate::graph::get_indexed_inputs(dag, *node_id).unwrap_or_default();
                        let input_tids = inputs.iter().map(|&idx| dag[idx].tensor_id).collect();
                        let output_tid = dag[*node_id].tensor_id;
                        (input_tids, output_tid)
                    }
                    ScheduledOp::Fused(fused) => match fused {
                        crate::scheduler::FusedOp::MatMulRelu { a, b, output } => (
                            vec![dag[*a].tensor_id, dag[*b].tensor_id],
                            dag[*output].tensor_id,
                        ),
                        crate::scheduler::FusedOp::MatMulAdd { a, b, bias, output } => (
                            vec![dag[*a].tensor_id, dag[*b].tensor_id, dag[*bias].tensor_id],
                            dag[*output].tensor_id,
                        ),
                        crate::scheduler::FusedOp::MatMulAddRelu { a, b, bias, output } => (
                            vec![dag[*a].tensor_id, dag[*b].tensor_id, dag[*bias].tensor_id],
                            dag[*output].tensor_id,
                        ),
                        crate::scheduler::FusedOp::ElementwiseChain { input, output, .. } => {
                            (vec![dag[*input].tensor_id], dag[*output].tensor_id)
                        }
                    },
                }
            };

            let num_ops = schedule.len();
            let mut in_degrees = Vec::with_capacity(num_ops);
            let mut tensor_to_consumers: HashMap<TensorId, Vec<usize>> = HashMap::new();
            let mut remaining_consumers = HashMap::new();

            // Precompute tensor to consumer mapping and count consumers
            for (op_idx, scheduled_op) in schedule.iter().enumerate() {
                let (inputs, _) = get_op_io(scheduled_op);
                for &in_tid in &inputs {
                    tensor_to_consumers.entry(in_tid).or_default().push(op_idx);
                }
            }

            // Initialize remaining_consumers for all input/intermediate tensors
            for (&tid, consumers) in &tensor_to_consumers {
                remaining_consumers.insert(tid, AtomicUsize::new(consumers.len()));
            }

            // Build a separate map of output_tid → consumer steps, but only
            // for tensor IDs that were NOT already in the registry.  These
            // are the "real" dependencies — steps that produce intermediate
            // results.  Steps whose inputs are all pre-registered have
            // in-degree 0 and can start immediately.
            let mut tensor_to_pending: HashMap<TensorId, Vec<usize>> = HashMap::new();
            for (op_idx, scheduled_op) in schedule.iter().enumerate() {
                let (inputs, _) = get_op_io(scheduled_op);
                for &in_tid in &inputs {
                    if !registry.contains(in_tid) {
                        tensor_to_pending.entry(in_tid).or_default().push(op_idx);
                    }
                }
            }

            // Initialize in-degrees: each op starts with as many missing
            // dependencies as it has unregistered input tensors.
            for scheduled_op in &schedule {
                let (inputs, _) = get_op_io(scheduled_op);
                let mut degree = 0;
                for &in_tid in &inputs {
                    if !registry.contains(in_tid) {
                        degree += 1;
                    }
                }
                in_degrees.push(AtomicUsize::new(degree));
            }

            let run_op = |op_idx: usize| -> Result<(), Error> {
                let scheduled_op = &schedule[op_idx];
                let step_device = {
                    let node_id = match scheduled_op {
                        ScheduledOp::Plain(node_id, _) => *node_id,
                        ScheduledOp::Fused(fused) => match fused {
                            crate::scheduler::FusedOp::MatMulRelu { output, .. } => *output,
                            crate::scheduler::FusedOp::MatMulAdd { output, .. } => *output,
                            crate::scheduler::FusedOp::MatMulAddRelu { output, .. } => *output,
                            crate::scheduler::FusedOp::ElementwiseChain { output, .. } => *output,
                        },
                    };
                    let tid = dag[node_id].tensor_id;
                    target.graph.get_device(tid, target_device)
                };

                // A. Pin all inputs
                let (inputs, _) = get_op_io(scheduled_op);
                for &in_tid in &inputs {
                    registry.pin(in_tid);
                }

                // B. Prefetch any input tensors scheduled before this step (only on GPU)
                if step_device == Device::Wgpu {
                    if let Some(tids) = prefetch_plan.prefetch_before.get(&op_idx) {
                        let wgpu_backend = WgpuBackend::get_or_init()?;
                        for &tid in tids {
                            registry.ensure_gpu(
                                tid,
                                wgpu_backend.device(),
                                wgpu_backend.queue(),
                            )?;
                        }
                    }
                }

                // C. Execute step
                let res = if step_device == Device::Wgpu {
                    execute_gpu_step(op_idx, scheduled_op, &registry, dag)
                } else {
                    execute_cpu_step(op_idx, scheduled_op, &registry, dag, &cpu_backend)
                };

                // D. Unpin inputs
                for &in_tid in &inputs {
                    registry.unpin(in_tid);
                }

                res
            };

            let (tx, rx) = mpsc::channel::<(usize, Result<(), Error>)>();
            let shared_err = Mutex::new(None);
            let cancelled = std::sync::atomic::AtomicBool::new(false);
            let rx = Arc::new(Mutex::new(rx));

            {
                rayon::scope(|s| {
                    // Spawn initially ready tasks
                    for (op_idx, deg) in in_degrees.iter().enumerate() {
                        if deg.load(Ordering::SeqCst) == 0 {
                            let tx = tx.clone();
                            let cancelled_ref = &cancelled;
                            s.spawn(move |_| {
                                if !cancelled_ref.load(Ordering::SeqCst) {
                                    let res = run_op(op_idx);
                                    let _ = tx.send((op_idx, res));
                                } else {
                                    let _ = tx.send((op_idx, Ok(())));
                                }
                            });
                        }
                    }

                    // Coordinator loop
                    let mut completed_tasks = 0;
                    while completed_tasks < num_ops {
                        let (op_idx, res) = {
                            let rx_lock = rx
                                .lock()
                                .expect("Runtime coordinator channel lock poisoned");
                            match rx_lock.recv() {
                                Ok(val) => val,
                                Err(_) => break, // Channel closed
                            }
                        };
                        completed_tasks += 1;

                        match res {
                            Ok(()) => {
                                if cancelled.load(Ordering::SeqCst) {
                                    continue;
                                }
                                // Decrement consumer ref counts and free if 0
                                let scheduled_op = &schedule[op_idx];
                                let (inputs, output_tid) = get_op_io(scheduled_op);
                                for &in_tid in &inputs {
                                    if in_tid != target_tid {
                                        if let Some(ref_cnt) = remaining_consumers.get(&in_tid) {
                                            if ref_cnt.fetch_sub(1, Ordering::SeqCst) == 1 {
                                                registry.free(in_tid);
                                            }
                                        }
                                    }
                                }

                                // Resolve dependencies for consumer tasks
                                // Only tensor IDs that were NOT pre-registered
                                // contribute to in-degree, so only those producers
                                // can unblock a consumer.
                                if let Some(consumers) = tensor_to_pending.get(&output_tid) {
                                    for &consumer_op_idx in consumers {
                                        let prev = in_degrees[consumer_op_idx]
                                            .fetch_sub(1, Ordering::SeqCst);
                                        if prev == 1 {
                                            let tx = tx.clone();
                                            let cancelled_ref = &cancelled;
                                            s.spawn(move |_| {
                                                if !cancelled_ref.load(Ordering::SeqCst) {
                                                    let res = run_op(consumer_op_idx);
                                                    let _ = tx.send((consumer_op_idx, res));
                                                } else {
                                                    let _ = tx.send((consumer_op_idx, Ok(())));
                                                }
                                            });
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                cancelled.store(true, Ordering::SeqCst);
                                let mut guard =
                                    shared_err.lock().expect("Shared error lock poisoned");
                                if guard.is_none() {
                                    *guard = Some(e);
                                }
                            }
                        }
                    }
                });
            }

            if let Some(e) = shared_err.into_inner().unwrap() {
                return Err(e);
            }

            // 8. Ensure output tensor is read back to CPU (skip if compute-only)
            let final_data = if skip_readback {
                Vec::new()
            } else if target_device == Device::Wgpu {
                let wgpu_backend = WgpuBackend::get_or_init()?;
                registry.ensure_cpu(target_tid, Some(wgpu_backend.device()))?
            } else {
                registry.ensure_cpu(target_tid, None)?
            };

            let final_shape = dag[target.id].shape.clone();
            Ok((final_data, final_shape))
        }?;

        // 9. Update diagnostics on target graph
        if target_device == Device::Wgpu {
            let mut inner = target.graph.inner.write().unwrap();
            inner.peak_gpu_bytes = registry.peak_gpu_bytes();
            inner.upload_count = registry.upload_count();
            inner.eviction_count = registry.eviction_count();
        } else {
            let mut inner = target.graph.inner.write().unwrap();
            inner.peak_gpu_bytes = 0;
            inner.upload_count = 0;
            inner.eviction_count = 0;
        }

        if skip_readback {
            Ok(Tensor::new(vec![], Shape::new(vec![0])))
        } else {
            Ok(Tensor::new(final_data, final_shape))
        }
    }

    /// Dispatch step execution to the host CPU backend.
    fn execute_cpu_step(
        op_idx: usize,
        scheduled_op: &ScheduledOp,
        registry: &BufferRegistry,
        dag: &petgraph::graph::DiGraph<Node, Edge>,
        backend: &dyn Backend,
    ) -> Result<(), Error> {
        match scheduled_op {
            ScheduledOp::Plain(node_id, op) => {
                let output_tid = dag[*node_id].tensor_id;
                let output_shape = dag[*node_id].shape.clone();

                match op {
                    Op::Input(_) => {
                        registry.touch(output_tid, op_idx)?;
                    }
                    _ => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let mut input_tensors = Vec::new();
                        for &in_node in &inputs {
                            let in_tid = dag[in_node].tensor_id;
                            let in_data = registry.ensure_cpu(in_tid, None)?;
                            registry.touch(in_tid, op_idx)?;
                            input_tensors.push(Tensor::new(in_data, dag[in_node].shape.clone()));
                        }
                        let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
                        let out_tensor = backend.execute(op, &input_refs)?;
                        registry.register_cpu(
                            output_tid,
                            out_tensor.data_raw().to_f32(),
                            output_shape,
                            dag[*node_id].dtype,
                        );
                        registry.touch(output_tid, op_idx)?;
                    }
                }
            }
            ScheduledOp::Fused(fused) => match fused {
                crate::scheduler::FusedOp::MatMulRelu { a, b, output } => {
                    let a_tid = dag[*a].tensor_id;
                    let b_tid = dag[*b].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let a_data = registry.ensure_cpu(a_tid, None)?;
                    let b_data = registry.ensure_cpu(b_tid, None)?;
                    registry.touch(a_tid, op_idx)?;
                    registry.touch(b_tid, op_idx)?;

                    let a_tensor = Tensor::new(a_data, dag[*a].shape.clone());
                    let b_tensor = Tensor::new(b_data, dag[*b].shape.clone());

                    let out_tensor = backend.execute_matmul_relu(&a_tensor, &b_tensor)?;
                    registry.register_cpu(
                        output_tid,
                        out_tensor.data_raw().to_f32(),
                        output_shape,
                        dag[*output].dtype,
                    );
                    registry.touch(output_tid, op_idx)?;
                }
                crate::scheduler::FusedOp::MatMulAdd { a, b, bias, output } => {
                    let a_tid = dag[*a].tensor_id;
                    let b_tid = dag[*b].tensor_id;
                    let bias_tid = dag[*bias].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let a_data = registry.ensure_cpu(a_tid, None)?;
                    let b_data = registry.ensure_cpu(b_tid, None)?;
                    let bias_data = registry.ensure_cpu(bias_tid, None)?;
                    registry.touch(a_tid, op_idx)?;
                    registry.touch(b_tid, op_idx)?;
                    registry.touch(bias_tid, op_idx)?;

                    let a_tensor = Tensor::new(a_data, dag[*a].shape.clone());
                    let b_tensor = Tensor::new(b_data, dag[*b].shape.clone());
                    let bias_tensor = Tensor::new(bias_data, dag[*bias].shape.clone());

                    let out_tensor =
                        backend.execute_matmul_add(&a_tensor, &b_tensor, &bias_tensor)?;
                    registry.register_cpu(
                        output_tid,
                        out_tensor.data_raw().to_f32(),
                        output_shape,
                        dag[*output].dtype,
                    );
                    registry.touch(output_tid, op_idx)?;
                }
                crate::scheduler::FusedOp::MatMulAddRelu { a, b, bias, output } => {
                    let a_tid = dag[*a].tensor_id;
                    let b_tid = dag[*b].tensor_id;
                    let bias_tid = dag[*bias].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let a_data = registry.ensure_cpu(a_tid, None)?;
                    let b_data = registry.ensure_cpu(b_tid, None)?;
                    let bias_data = registry.ensure_cpu(bias_tid, None)?;
                    registry.touch(a_tid, op_idx)?;
                    registry.touch(b_tid, op_idx)?;
                    registry.touch(bias_tid, op_idx)?;

                    let a_tensor = Tensor::new(a_data, dag[*a].shape.clone());
                    let b_tensor = Tensor::new(b_data, dag[*b].shape.clone());
                    let bias_tensor = Tensor::new(bias_data, dag[*bias].shape.clone());

                    let out_tensor =
                        backend.execute_matmul_add_relu(&a_tensor, &b_tensor, &bias_tensor)?;
                    registry.register_cpu(
                        output_tid,
                        out_tensor.data_raw().to_f32(),
                        output_shape,
                        dag[*output].dtype,
                    );
                    registry.touch(output_tid, op_idx)?;
                }
                crate::scheduler::FusedOp::ElementwiseChain { input, ops, output } => {
                    let input_tid = dag[*input].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let input_data = registry.ensure_cpu(input_tid, None)?;
                    registry.touch(input_tid, op_idx)?;
                    let input_shape = dag[*input].shape.clone();

                    let mut expr = crate::codegen::ast::Expr::Input(0);
                    for op in ops {
                        expr = match op {
                            Op::Relu => crate::codegen::ast::Expr::Relu(Box::new(expr)),
                            Op::Tanh => crate::codegen::ast::Expr::Tanh(Box::new(expr)),
                            Op::Sigmoid => crate::codegen::ast::Expr::Sigmoid(Box::new(expr)),
                            Op::Exp => crate::codegen::ast::Expr::Exp(Box::new(expr)),
                            Op::Sqrt => crate::codegen::ast::Expr::Sqrt(Box::new(expr)),
                            Op::Neg => crate::codegen::ast::Expr::Neg(Box::new(expr)),
                            Op::Step => crate::codegen::ast::Expr::Step(Box::new(expr)),
                            _ => {
                                return Err(Error::ExecutionError(format!(
                                    "Unsupported op in CPU ElementwiseChain: {:?}",
                                    op
                                )))
                            }
                        };
                    }

                    let mut out_data = vec![0.0; output_shape.num_elements()];
                    for (i, out) in out_data.iter_mut().enumerate() {
                        *out = crate::codegen::ast::evaluate_ast(
                            &expr,
                            &[&input_data],
                            std::slice::from_ref(&input_shape),
                            &output_shape,
                            i,
                        );
                    }

                    registry.register_cpu(output_tid, out_data, output_shape, dag[*output].dtype);
                    registry.touch(output_tid, op_idx)?;
                }
            },
        }
        Ok(())
    }

    /// Dispatch step execution to the GPU using the WGPU/Metal backend.
    fn execute_gpu_step(
        op_idx: usize,
        scheduled_op: &ScheduledOp,
        registry: &BufferRegistry,
        dag: &petgraph::graph::DiGraph<Node, Edge>,
    ) -> Result<(), Error> {
        let wgpu_backend = WgpuBackend::get_or_init()?;
        let wgpu_device = wgpu_backend.device();
        let queue = wgpu_backend.queue();

        match scheduled_op {
            ScheduledOp::Plain(node_id, op) => {
                let output_tid = dag[*node_id].tensor_id;
                let output_shape = dag[*node_id].shape.clone();

                match op {
                    Op::Input(_) => {
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::MatMul => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        if inputs.len() != 2 {
                            return Err(Error::ExecutionError(
                                "GPU MatMul requires exactly 2 inputs".to_string(),
                            ));
                        }
                        let lhs_tid = dag[inputs[0]].tensor_id;
                        let rhs_tid = dag[inputs[1]].tensor_id;

                        let lhs_buf = registry.ensure_gpu(lhs_tid, wgpu_device, queue)?;
                        let rhs_buf = registry.ensure_gpu(rhs_tid, wgpu_device, queue)?;
                        registry.touch(lhs_tid, op_idx)?;
                        registry.touch(rhs_tid, op_idx)?;

                        let lhs_shape = dag[inputs[0]].shape.dims();
                        let rhs_shape = dag[inputs[1]].shape.dims();
                        let m = lhs_shape[0] as u32;
                        let k = lhs_shape[1] as u32;
                        let n = rhs_shape[1] as u32;

                        let output_buf =
                            wgpu_backend.execute_matmul_buffers(&lhs_buf, &rhs_buf, m, n, k)?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::Transpose => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        let dims = dag[inputs[0]].shape.dims();
                        let output_buf = wgpu_backend.execute_transpose_buffers(
                            &input_buf,
                            dims[0] as u32,
                            dims[1] as u32,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::SumAll => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        let output_buf = wgpu_backend.execute_sum_all_buffers(&input_buf)?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::SumDim { axis } => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        let output_buf = wgpu_backend.execute_sum_dim_buffers(
                            &input_buf,
                            &dag[inputs[0]].shape,
                            *axis,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::Reshape { .. } => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        registry.register_gpu(
                            output_tid,
                            input_buf,
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::Softmax => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        let output_buf = wgpu_backend
                            .execute_softmax_buffers(&input_buf, &dag[inputs[0]].shape)?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::SoftmaxGrad => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let x_tid = dag[inputs[0]].tensor_id;
                        let grad_tid = dag[inputs[1]].tensor_id;

                        let x_buf = registry.ensure_gpu(x_tid, wgpu_device, queue)?;
                        let grad_buf = registry.ensure_gpu(grad_tid, wgpu_device, queue)?;
                        registry.touch(x_tid, op_idx)?;
                        registry.touch(grad_tid, op_idx)?;

                        let output_buf = wgpu_backend.execute_softmax_grad_buffers(
                            &x_buf,
                            &grad_buf,
                            &dag[inputs[0]].shape,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::Concat { axis } => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let mut input_bufs = Vec::new();
                        let mut input_shapes = Vec::new();
                        for &in_node in &inputs {
                            let in_tid = dag[in_node].tensor_id;
                            let in_buf = registry.ensure_gpu(in_tid, wgpu_device, queue)?;
                            registry.touch(in_tid, op_idx)?;
                            input_bufs.push(in_buf);
                            input_shapes.push(dag[in_node].shape.clone());
                        }
                        let input_buf_refs: Vec<&wgpu::Buffer> =
                            input_bufs.iter().map(|b| b.as_ref()).collect();
                        let output_buf = wgpu_backend.execute_concat_buffers(
                            &input_buf_refs,
                            &input_shapes,
                            *axis,
                            &output_shape,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::LayerNorm { epsilon } => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let x_tid = dag[inputs[0]].tensor_id;
                        let weight_tid = dag[inputs[1]].tensor_id;
                        let bias_tid = dag[inputs[2]].tensor_id;

                        let x_buf = registry.ensure_gpu(x_tid, wgpu_device, queue)?;
                        let w_buf = registry.ensure_gpu(weight_tid, wgpu_device, queue)?;
                        let b_buf = registry.ensure_gpu(bias_tid, wgpu_device, queue)?;

                        registry.touch(x_tid, op_idx)?;
                        registry.touch(weight_tid, op_idx)?;
                        registry.touch(bias_tid, op_idx)?;

                        let output_buf = wgpu_backend.execute_layernorm_buffers(
                            &x_buf,
                            &w_buf,
                            &b_buf,
                            &dag[inputs[0]].shape,
                            *epsilon,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::Conv2d { stride, padding } => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let x_tid = dag[inputs[0]].tensor_id;
                        let weight_tid = dag[inputs[1]].tensor_id;

                        let x_buf = registry.ensure_gpu(x_tid, wgpu_device, queue)?;
                        let w_buf = registry.ensure_gpu(weight_tid, wgpu_device, queue)?;

                        registry.touch(x_tid, op_idx)?;
                        registry.touch(weight_tid, op_idx)?;

                        let bias_buf = if inputs.len() == 3 {
                            let bias_tid = dag[inputs[2]].tensor_id;
                            let b_buf = registry.ensure_gpu(bias_tid, wgpu_device, queue)?;
                            registry.touch(bias_tid, op_idx)?;
                            Some(b_buf)
                        } else {
                            None
                        };

                        let bias_buf_ref = bias_buf.as_ref().map(|b| b.as_ref());

                        let output_buf = wgpu_backend.execute_conv2d_buffers(
                            &x_buf,
                            &w_buf,
                            bias_buf_ref,
                            &dag[inputs[0]].shape,
                            &dag[inputs[1]].shape,
                            *stride,
                            *padding,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::BatchedMatMul => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        if inputs.len() != 2 {
                            return Err(Error::ExecutionError(
                                "GPU BatchedMatMul requires exactly 2 inputs".to_string(),
                            ));
                        }
                        let lhs_tid = dag[inputs[0]].tensor_id;
                        let rhs_tid = dag[inputs[1]].tensor_id;

                        let lhs_buf = registry.ensure_gpu(lhs_tid, wgpu_device, queue)?;
                        let rhs_buf = registry.ensure_gpu(rhs_tid, wgpu_device, queue)?;
                        registry.touch(lhs_tid, op_idx)?;
                        registry.touch(rhs_tid, op_idx)?;

                        let lhs_shape = dag[inputs[0]].shape.dims();
                        let rhs_shape = dag[inputs[1]].shape.dims();
                        let b = lhs_shape[0] as u32;
                        let m = lhs_shape[1] as u32;
                        let k = lhs_shape[2] as u32;
                        let n = rhs_shape[2] as u32;

                        let output_buf = wgpu_backend
                            .execute_batched_matmul_buffers(&lhs_buf, &rhs_buf, b, m, n, k)?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::BatchedTranspose => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        let dims = dag[inputs[0]].shape.dims();
                        let b = dims[0] as u32;
                        let m = dims[1] as u32;
                        let n = dims[2] as u32;

                        let output_buf =
                            wgpu_backend.execute_batched_transpose_buffers(&input_buf, b, m, n)?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::MaxPool2d {
                        pool_size,
                        stride,
                        padding,
                    } => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        let in_dims = dag[inputs[0]].shape.dims();
                        let out_dims = output_shape.dims();

                        let n = in_dims[0] as u32;
                        let c = in_dims[1] as u32;
                        let h = in_dims[2] as u32;
                        let w = in_dims[3] as u32;

                        let out_h = out_dims[2] as u32;
                        let out_w = out_dims[3] as u32;

                        let output_buf = wgpu_backend.execute_max_pool2d_buffers(
                            &input_buf,
                            n,
                            c,
                            h,
                            w,
                            out_h,
                            out_w,
                            *pool_size as u32,
                            *pool_size as u32,
                            *stride as u32,
                            *stride as u32,
                            *padding as u32,
                            *padding as u32,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::AvgPool2d {
                        pool_size,
                        stride,
                        padding,
                    } => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let input_tid = dag[inputs[0]].tensor_id;
                        let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                        registry.touch(input_tid, op_idx)?;

                        let in_dims = dag[inputs[0]].shape.dims();
                        let out_dims = output_shape.dims();

                        let n = in_dims[0] as u32;
                        let c = in_dims[1] as u32;
                        let h = in_dims[2] as u32;
                        let w = in_dims[3] as u32;

                        let out_h = out_dims[2] as u32;
                        let out_w = out_dims[3] as u32;

                        let output_buf = wgpu_backend.execute_avg_pool2d_buffers(
                            &input_buf,
                            n,
                            c,
                            h,
                            w,
                            out_h,
                            out_w,
                            *pool_size as u32,
                            *pool_size as u32,
                            *stride as u32,
                            *stride as u32,
                            *padding as u32,
                            *padding as u32,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::MaxPool2dGrad {
                        pool_size,
                        stride,
                        padding,
                    } => {
                        // dy  = inputs[0]  [N, C, out_H, out_W]
                        // x   = inputs[1]  [N, C, H, W]
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        if inputs.len() != 2 {
                            return Err(Error::ExecutionError(
                                "MaxPool2dGrad requires 2 inputs".to_string(),
                            ));
                        }
                        let dy_tid = dag[inputs[0]].tensor_id;
                        let x_tid = dag[inputs[1]].tensor_id;
                        let dy_buf = registry.ensure_gpu(dy_tid, wgpu_device, queue)?;
                        let x_buf = registry.ensure_gpu(x_tid, wgpu_device, queue)?;
                        registry.touch(dy_tid, op_idx)?;
                        registry.touch(x_tid, op_idx)?;

                        let x_dims = dag[inputs[1]].shape.dims();
                        let dy_dims = dag[inputs[0]].shape.dims();
                        let n = x_dims[0] as u32;
                        let c = x_dims[1] as u32;
                        let h = x_dims[2] as u32;
                        let w = x_dims[3] as u32;
                        let oh = dy_dims[2] as u32;
                        let ow = dy_dims[3] as u32;

                        let out_buf = wgpu_backend.execute_max_pool2d_grad_buffers(
                            &dy_buf,
                            &x_buf,
                            n,
                            c,
                            h,
                            w,
                            oh,
                            ow,
                            *pool_size as u32,
                            *pool_size as u32,
                            *stride as u32,
                            *stride as u32,
                            *padding as u32,
                            *padding as u32,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(out_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::AvgPool2dGrad {
                        pool_size,
                        stride,
                        padding,
                    } => {
                        // dy  = inputs[0]  [N, C, out_H, out_W]
                        // x   = inputs[1]  [N, C, H, W]   (only used for shape)
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        if inputs.len() != 2 {
                            return Err(Error::ExecutionError(
                                "AvgPool2dGrad requires 2 inputs".to_string(),
                            ));
                        }
                        let dy_tid = dag[inputs[0]].tensor_id;
                        let dy_buf = registry.ensure_gpu(dy_tid, wgpu_device, queue)?;
                        registry.touch(dy_tid, op_idx)?;

                        let x_dims = dag[inputs[1]].shape.dims();
                        let dy_dims = dag[inputs[0]].shape.dims();
                        let n = x_dims[0] as u32;
                        let c = x_dims[1] as u32;
                        let h = x_dims[2] as u32;
                        let w = x_dims[3] as u32;
                        let oh = dy_dims[2] as u32;
                        let ow = dy_dims[3] as u32;

                        let out_buf = wgpu_backend.execute_avg_pool2d_grad_buffers(
                            &dy_buf,
                            n,
                            c,
                            h,
                            w,
                            oh,
                            ow,
                            *pool_size as u32,
                            *pool_size as u32,
                            *stride as u32,
                            *stride as u32,
                            *padding as u32,
                            *padding as u32,
                        )?;
                        registry.register_gpu(
                            output_tid,
                            Arc::new(out_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    Op::Attention { .. }
                    | Op::AttentionGradQ { .. }
                    | Op::AttentionGradK { .. }
                    | Op::AttentionGradV { .. }
                    | Op::CausalAttention { .. }
                    | Op::MultiHeadAttention { .. }
                    | Op::FlashAttention { .. }
                    | Op::CastF32ToF16
                    | Op::CastF16ToF32
                    | Op::CastF32ToBF16
                    | Op::CastBF16ToF32 => {
                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let mut input_tensors = Vec::new();
                        for &in_node in &inputs {
                            let in_tid = dag[in_node].tensor_id;
                            let in_data = registry.ensure_cpu(in_tid, Some(wgpu_device))?;
                            registry.touch(in_tid, op_idx)?;
                            input_tensors.push(Tensor::new(in_data, dag[in_node].shape.clone()));
                        }
                        let input_refs: Vec<&Tensor> = input_tensors.iter().collect();

                        let cpu_backend = CpuBackend::new();
                        let out_tensor = cpu_backend.execute(op, &input_refs)?;

                        let output_buf = wgpu_backend.create_buffer_with_data(
                            out_tensor.data(),
                            wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                        );

                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                    _ => {
                        let expr = match op {
                            Op::Relu => crate::codegen::ast::Expr::Relu(Box::new(
                                crate::codegen::ast::Expr::Input(0),
                            )),
                            Op::Tanh => crate::codegen::ast::Expr::Tanh(Box::new(
                                crate::codegen::ast::Expr::Input(0),
                            )),
                            Op::Sigmoid => crate::codegen::ast::Expr::Sigmoid(Box::new(
                                crate::codegen::ast::Expr::Input(0),
                            )),
                            Op::Exp => crate::codegen::ast::Expr::Exp(Box::new(
                                crate::codegen::ast::Expr::Input(0),
                            )),
                            Op::Sqrt => crate::codegen::ast::Expr::Sqrt(Box::new(
                                crate::codegen::ast::Expr::Input(0),
                            )),
                            Op::Neg => crate::codegen::ast::Expr::Neg(Box::new(
                                crate::codegen::ast::Expr::Input(0),
                            )),
                            Op::Step => crate::codegen::ast::Expr::Step(Box::new(
                                crate::codegen::ast::Expr::Input(0),
                            )),
                            Op::Add => crate::codegen::ast::Expr::Add(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            Op::Sub => crate::codegen::ast::Expr::Sub(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            Op::Mul => crate::codegen::ast::Expr::Mul(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            Op::Div => crate::codegen::ast::Expr::Div(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            Op::BroadcastAdd { .. } => crate::codegen::ast::Expr::Add(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            Op::BroadcastMul { .. } => crate::codegen::ast::Expr::Mul(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            Op::BroadcastSub { .. } => crate::codegen::ast::Expr::Sub(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            Op::BroadcastDiv { .. } => crate::codegen::ast::Expr::Div(
                                Box::new(crate::codegen::ast::Expr::Input(0)),
                                Box::new(crate::codegen::ast::Expr::Input(1)),
                            ),
                            _ => {
                                return Err(Error::ExecutionError(format!(
                                    "Unsupported GPU op in runtime: {:?}",
                                    op
                                )))
                            }
                        };

                        let inputs = get_indexed_inputs(dag, *node_id)?;
                        let mut input_buffers = Vec::new();
                        let mut input_shapes = Vec::new();
                        for &in_node in &inputs {
                            let in_tid = dag[in_node].tensor_id;
                            let in_buf = registry.ensure_gpu(in_tid, wgpu_device, queue)?;
                            registry.touch(in_tid, op_idx)?;
                            input_buffers.push(in_buf);
                            input_shapes.push(dag[in_node].shape.clone());
                        }

                        let input_buffer_refs: Vec<&wgpu::Buffer> =
                            input_buffers.iter().map(|b| b.as_ref()).collect();
                        let output_buf = wgpu_backend.execute_ast_buffers(
                            &expr,
                            &input_buffer_refs,
                            &input_shapes,
                            &output_shape,
                        )?;

                        registry.register_gpu(
                            output_tid,
                            Arc::new(output_buf),
                            output_shape,
                            dag[*node_id].dtype,
                        )?;
                        registry.touch(output_tid, op_idx)?;
                    }
                }
            }
            ScheduledOp::Fused(fused) => match fused {
                crate::scheduler::FusedOp::MatMulRelu { a, b, output } => {
                    let a_tid = dag[*a].tensor_id;
                    let b_tid = dag[*b].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let a_buf = registry.ensure_gpu(a_tid, wgpu_device, queue)?;
                    let b_buf = registry.ensure_gpu(b_tid, wgpu_device, queue)?;
                    registry.touch(a_tid, op_idx)?;
                    registry.touch(b_tid, op_idx)?;

                    let lhs_shape = dag[*a].shape.dims();
                    let rhs_shape = dag[*b].shape.dims();
                    let m = lhs_shape[0] as u32;
                    let k = lhs_shape[1] as u32;
                    let n = rhs_shape[1] as u32;

                    let output_buf =
                        wgpu_backend.execute_matmul_relu_buffers(&a_buf, &b_buf, m, n, k)?;
                    registry.register_gpu(
                        output_tid,
                        Arc::new(output_buf),
                        output_shape,
                        dag[*output].dtype,
                    )?;
                    registry.touch(output_tid, op_idx)?;
                }
                crate::scheduler::FusedOp::MatMulAdd { a, b, bias, output } => {
                    let a_tid = dag[*a].tensor_id;
                    let b_tid = dag[*b].tensor_id;
                    let bias_tid = dag[*bias].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let a_buf = registry.ensure_gpu(a_tid, wgpu_device, queue)?;
                    let b_buf = registry.ensure_gpu(b_tid, wgpu_device, queue)?;
                    let bias_buf = registry.ensure_gpu(bias_tid, wgpu_device, queue)?;
                    registry.touch(a_tid, op_idx)?;
                    registry.touch(b_tid, op_idx)?;
                    registry.touch(bias_tid, op_idx)?;

                    let lhs_shape = dag[*a].shape.dims();
                    let rhs_shape = dag[*b].shape.dims();
                    let m = lhs_shape[0] as u32;
                    let k = lhs_shape[1] as u32;
                    let n = rhs_shape[1] as u32;

                    let output_buf = wgpu_backend
                        .execute_matmul_add_buffers(&a_buf, &b_buf, &bias_buf, m, n, k)?;
                    registry.register_gpu(
                        output_tid,
                        Arc::new(output_buf),
                        output_shape,
                        dag[*output].dtype,
                    )?;
                    registry.touch(output_tid, op_idx)?;
                }
                crate::scheduler::FusedOp::MatMulAddRelu { a, b, bias, output } => {
                    let a_tid = dag[*a].tensor_id;
                    let b_tid = dag[*b].tensor_id;
                    let bias_tid = dag[*bias].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let a_buf = registry.ensure_gpu(a_tid, wgpu_device, queue)?;
                    let b_buf = registry.ensure_gpu(b_tid, wgpu_device, queue)?;
                    let bias_buf = registry.ensure_gpu(bias_tid, wgpu_device, queue)?;
                    registry.touch(a_tid, op_idx)?;
                    registry.touch(b_tid, op_idx)?;
                    registry.touch(bias_tid, op_idx)?;

                    let lhs_shape = dag[*a].shape.dims();
                    let rhs_shape = dag[*b].shape.dims();
                    let m = lhs_shape[0] as u32;
                    let k = lhs_shape[1] as u32;
                    let n = rhs_shape[1] as u32;

                    let output_buf = wgpu_backend
                        .execute_matmul_add_relu_buffers(&a_buf, &b_buf, &bias_buf, m, n, k)?;
                    registry.register_gpu(
                        output_tid,
                        Arc::new(output_buf),
                        output_shape,
                        dag[*output].dtype,
                    )?;
                    registry.touch(output_tid, op_idx)?;
                }
                crate::scheduler::FusedOp::ElementwiseChain { input, ops, output } => {
                    let input_tid = dag[*input].tensor_id;
                    let output_tid = dag[*output].tensor_id;
                    let output_shape = dag[*output].shape.clone();

                    let input_buf = registry.ensure_gpu(input_tid, wgpu_device, queue)?;
                    registry.touch(input_tid, op_idx)?;

                    let input_shapes = vec![dag[*input].shape.clone()];
                    let input_buffer_refs = vec![input_buf.as_ref()];

                    let mut expr = crate::codegen::ast::Expr::Input(0);
                    for op in ops {
                        expr = match op {
                            Op::Relu => crate::codegen::ast::Expr::Relu(Box::new(expr)),
                            Op::Tanh => crate::codegen::ast::Expr::Tanh(Box::new(expr)),
                            Op::Sigmoid => crate::codegen::ast::Expr::Sigmoid(Box::new(expr)),
                            Op::Exp => crate::codegen::ast::Expr::Exp(Box::new(expr)),
                            Op::Sqrt => crate::codegen::ast::Expr::Sqrt(Box::new(expr)),
                            Op::Neg => crate::codegen::ast::Expr::Neg(Box::new(expr)),
                            Op::Step => crate::codegen::ast::Expr::Step(Box::new(expr)),
                            _ => {
                                return Err(Error::ExecutionError(format!(
                                    "Unsupported op in GPU ElementwiseChain: {:?}",
                                    op
                                )))
                            }
                        };
                    }

                    let output_buf = wgpu_backend.execute_ast_buffers(
                        &expr,
                        &input_buffer_refs,
                        &input_shapes,
                        &output_shape,
                    )?;
                    registry.register_gpu(
                        output_tid,
                        Arc::new(output_buf),
                        output_shape,
                        dag[*output].dtype,
                    )?;
                    registry.touch(output_tid, op_idx)?;
                }
            },
        }
        Ok(())
    }
}
pub use runtime_mod::{execute, execute_no_readback};
