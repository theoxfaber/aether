/// Deferred-compute DAG — the primary user-facing API for building and
/// executing computation graphs.
///
/// # Overview
///
/// 1. Create a [`Graph`] with [`Graph::new`].
/// 2. Add tensors with [`Graph::tensor`] — returns a [`GraphTensor`].
/// 3. Chain operations (`.add()`, `.matmul()`, `.relu()`, …).
/// 4. Call [`GraphTensor::run`] to execute, or [`Graph::compile`] then `run`.
///
/// Every operation is recorded as a [`Node`] in a [`DiGraph`] (petgraph).
/// Execution is deferred — no computation happens until `run()`.
///
/// # Autograd
///
/// Call `.backward()` on a scalar tensor to build the reverse-mode gradient
/// graph.  Retrieve gradients by tensor ID from the returned map.
///
/// # Compiler
///
/// [`Graph::compile`] runs six passes: simplify → CSE → DCE → constant_fold
/// → DCE → layout_optimize.  Set [`Graph::set_compile_on_run(true)`] to
/// auto-compile before every `run()`.
pub mod graph_mod {
    use crate::tensor::{AnyData, Dtype, Shape, Tensor, TensorId};
    use crate::Device;
    use crate::Error;
    use petgraph::graph::{DiGraph, NodeIndex};
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    macro_rules! check_assert {
        ($self:expr, $cond:expr, $msg:expr) => {
            if !$cond {
                return $self.record_error_and_dummy($msg.to_string());
            }
        };
    }

    /// The list of compute operations supported by the Aether graph compiler.
    #[derive(Debug, Clone)]
    pub enum Op {
        /// A leaf input tensor wrapping concrete host-side values.
        Input(Tensor),
        /// Standard 2D matrix multiplication (matmul).
        MatMul,
        /// Rectified Linear Unit activation function (relu).
        Relu,
        /// Elementwise addition.
        Add,
        /// Elementwise subtraction.
        Sub,
        /// Elementwise multiplication.
        Mul,
        /// Elementwise division.
        Div,
        /// Hyperbolic tangent activation function.
        Tanh,
        /// Sigmoid activation function.
        Sigmoid,
        /// Exponential function (e^x).
        Exp,
        /// Square root function.
        Sqrt,
        /// Negation (-x).
        Neg,
        /// Broadcasted addition for mismatched but broadcastable shapes.
        BroadcastAdd { input_shapes: Vec<Shape> },
        /// Broadcasted multiplication for mismatched but broadcastable shapes.
        BroadcastMul { input_shapes: Vec<Shape> },
        /// Broadcasted subtraction for mismatched but broadcastable shapes.
        BroadcastSub { input_shapes: Vec<Shape> },
        /// Broadcasted division for mismatched but broadcastable shapes.
        BroadcastDiv { input_shapes: Vec<Shape> },
        /// Swap dimensions of a 2D matrix.
        Transpose,
        /// Reduce the entire tensor to a single-element scalar `Shape([1])`.
        SumAll,
        /// Reduce along a specific axis by summing.
        SumDim { axis: usize },
        /// Reshape the tensor to a new shape with the same number of elements.
        Reshape { shape: Shape },
        /// Compute softmax along the last dimension of the tensor.
        Softmax,
        /// Internal backprop op: computes the gradient of softmax.
        SoftmaxGrad,
        /// Internal backprop op: computes the step function (x >= 0 ? 1 : 0) for Relu gradient.
        Step,
        /// Concatenate multiple tensors along an axis.
        Concat { axis: usize },
        /// Layer Normalization.
        LayerNorm { epsilon: f32 },
        /// 2D Convolution.
        Conv2d { stride: usize, padding: usize },
        /// Slice a tensor along an axis.
        Slice {
            axis: usize,
            start: usize,
            end: usize,
        },
        /// Gradient of Slice: embeds dy into zero tensor of shape x.
        SliceGrad {
            axis: usize,
            start: usize,
            end: usize,
        },
        /// Gradient of LayerNorm with respect to input x.
        LayerNormGradX { epsilon: f32 },
        /// Gradient of LayerNorm with respect to weight.
        LayerNormGradW { epsilon: f32 },
        /// Gradient of LayerNorm with respect to bias.
        LayerNormGradB,
        /// Gradient of Conv2d with respect to input x.
        Conv2dGradX { stride: usize, padding: usize },
        /// Gradient of Conv2d with respect to weight.
        Conv2dGradW { stride: usize, padding: usize },
        /// Gradient of Conv2d with respect to bias.
        Conv2dGradB,
        /// Batched 3D matrix multiplication [B, M, K] * [B, K, N] -> [B, M, N].
        BatchedMatMul,
        /// Batched transpose of the last two dimensions [B, M, N] -> [B, N, M].
        BatchedTranspose,
        /// 2D Max Pooling.
        MaxPool2d {
            pool_size: usize,
            stride: usize,
            padding: usize,
        },
        /// Gradient of MaxPool2d.
        MaxPool2dGrad {
            pool_size: usize,
            stride: usize,
            padding: usize,
        },
        /// 2D Average Pooling.
        AvgPool2d {
            pool_size: usize,
            stride: usize,
            padding: usize,
        },
        /// Gradient of AvgPool2d.
        AvgPool2dGrad {
            pool_size: usize,
            stride: usize,
            padding: usize,
        },
        /// Scaled Dot-Product Attention: Softmax(Q * K^T * scale) * V.
        Attention { scale: f32 },
        /// Gradient of Attention with respect to Q.
        AttentionGradQ { scale: f32 },
        /// Gradient of Attention with respect to K.
        AttentionGradK { scale: f32 },
        /// Gradient of Attention with respect to V.
        AttentionGradV { scale: f32 },
        /// Cast from F32 to F16.
        CastF32ToF16,
        /// Cast from F16 to F32.
        CastF16ToF32,
        /// Cast from F32 to BF16.
        CastF32ToBF16,
        /// Cast from BF16 to F32.
        CastBF16ToF32,
        /// Causal (masked) Attention: Softmax(Q * K^T * scale + mask) * V.
        CausalAttention { scale: f32, num_heads: usize },
        /// Multi-Head Attention.
        MultiHeadAttention { scale: f32, num_heads: usize },
        /// Flash Attention (tiled, O(1) memory w.r.t. sequence length).
        FlashAttention { scale: f32, causal: bool },
        /// Root Mean Square Normalization (RMSNorm).
        RmsNorm { epsilon: f32 },
    }

    /// Represents a computation node in the graph, containing the operation, shape, type, and unique tensor ID.
    #[derive(Debug, Clone)]
    pub struct Node {
        /// The operation this node executes.
        pub op: Op,
        /// Output tensor shape of this node.
        pub shape: Shape,
        /// Output tensor data type.
        pub dtype: Dtype,
        /// Unique identifier for tracking memory buffers associated with this node.
        pub tensor_id: TensorId,
    }

    /// Edges in the directed graph representing connections.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Edge {
        /// Left-hand side of a binary operation.
        BinaryLhs,
        /// Right-hand side of a binary operation.
        BinaryRhs,
        /// Input to a unary operation.
        Unary,
        /// Index of input for multi-input operations.
        Index(usize),
    }

    /// A compute DAG (Directed Acyclic Graph) containing the scheduling and structure of computational steps.
    #[derive(Clone)]
    pub struct Graph {
        pub(crate) inner: Arc<RwLock<GraphInner>>,
    }

    impl Default for Graph {
        fn default() -> Self {
            Self::new()
        }
    }

    pub(crate) struct GraphInner {
        pub(crate) dag: DiGraph<Node, Edge>,
        pub(crate) device_assignments: HashMap<TensorId, crate::Device>,
        pub(crate) enable_fusion: bool,
        pub(crate) compile_on_run: bool,
        pub(crate) gpu_byte_limit: usize,
        pub(crate) peak_gpu_bytes: usize,
        pub(crate) upload_count: usize,
        pub(crate) eviction_count: usize,
        pub(crate) error: Option<String>,
    }

    impl Graph {
        /// Create a new, empty compute graph.
        pub fn new() -> Self {
            Self {
                inner: Arc::new(RwLock::new(GraphInner {
                    dag: DiGraph::new(),
                    device_assignments: HashMap::new(),
                    enable_fusion: false,
                    compile_on_run: false,
                    gpu_byte_limit: 6 * 1024 * 1024 * 1024, // Default 6GB
                    peak_gpu_bytes: 0,
                    upload_count: 0,
                    eviction_count: 0,
                    error: None,
                })),
            }
        }

        /// Configure the graph to run the compiler optimisation passes before
        /// every `run()` call.  When false (the default), `run()` executes the
        /// graph as-is without any optimisation.
        pub fn set_compile_on_run(&self, enabled: bool) {
            let mut inner = self.inner.write().unwrap();
            inner.compile_on_run = enabled;
        }

        pub fn set_error(&self, msg: String) {
            let mut inner = self.inner.write().unwrap();
            if inner.error.is_none() {
                inner.error = Some(msg);
            }
        }

        pub fn get_error(&self) -> Option<String> {
            self.inner.read().unwrap().error.clone()
        }

        pub fn clear_error(&self) {
            self.inner.write().unwrap().error = None;
        }

        /// Get the assigned device of a tensor or return default_device.
        pub fn get_device(&self, id: TensorId, default_device: crate::Device) -> crate::Device {
            let inner = self.inner.read().unwrap();
            inner
                .device_assignments
                .get(&id)
                .cloned()
                .unwrap_or(default_device)
        }

        /// Toggle the optimization scheduler's operator fusion passes.
        pub fn enable_fusion(&self, enable: bool) {
            self.inner.write().unwrap().enable_fusion = enable;
        }

        /// Retrieve the soft limit on GPU resident memory bytes.
        pub fn gpu_byte_limit(&self) -> usize {
            self.inner.read().unwrap().gpu_byte_limit
        }

        /// Set the soft limit on GPU resident memory bytes before triggering LRU eviction.
        pub fn set_gpu_memory_limit(&mut self, bytes: usize) -> &mut Self {
            self.inner.write().unwrap().gpu_byte_limit = bytes;
            self
        }

        /// Get the peak GPU resident bytes allocated during execution.
        pub fn peak_gpu_bytes(&self) -> usize {
            self.inner.read().unwrap().peak_gpu_bytes
        }

        /// Get the total count of host-to-device buffer uploads.
        pub fn upload_count(&self) -> usize {
            self.inner.read().unwrap().upload_count
        }

        /// Get the total count of device-to-host LRU buffer evictions.
        pub fn eviction_count(&self) -> usize {
            self.inner.read().unwrap().eviction_count
        }

        /// Update the data of an input tensor node in the graph.
        pub fn update_input(&self, id: petgraph::prelude::NodeIndex, new_data: Vec<f32>) {
            let mut inner = self.inner.write().unwrap();
            if let Some(node) = inner.dag.node_weight_mut(id) {
                if let Op::Input(ref mut tensor) = node.op {
                    tensor.data = crate::tensor::AnyData::F32(new_data);
                }
            }
        }

        /// Add a concrete tensor to the graph as a leaf input node.
        pub fn tensor(&self, data: Vec<f32>, shape: Shape) -> GraphTensor {
            let tensor = Tensor::new(data, shape.clone());
            let node = Node {
                op: Op::Input(tensor.clone()),
                shape,
                dtype: Dtype::F32,
                tensor_id: tensor.id(),
            };
            let id = self.inner.write().unwrap().dag.add_node(node);
            GraphTensor {
                id,
                graph: self.clone(),
            }
        }

        /// Add a tensor with arbitrary data (AnyData) and dtype to the graph.
        pub fn tensor_with_data(&self, data: AnyData, shape: Shape, dtype: Dtype) -> GraphTensor {
            let tensor = Tensor::new_with_data(data, shape.clone(), dtype);
            let tensor_id = tensor.id();
            let node = Node {
                op: Op::Input(tensor),
                shape,
                dtype,
                tensor_id,
            };
            let id = self.inner.write().unwrap().dag.add_node(node);
            GraphTensor {
                id,
                graph: self.clone(),
            }
        }

        /// Returns all leaf input tensors currently registered in this graph.
        pub fn input_tensors(&self) -> Vec<Tensor> {
            let inner = self.inner.read().unwrap();
            let mut inputs = Vec::new();
            for idx in inner.dag.node_indices() {
                if let Op::Input(tensor) = &inner.dag[idx].op {
                    inputs.push(tensor.clone());
                }
            }
            inputs
        }

        /// Compile the graph with optimization passes (constant folding, DCE, CSE, simplification).
        /// This modifies the graph in-place. Call before `run()` to get optimized execution.
        pub fn compile(&self) -> Result<(), Error> {
            let _compiled = crate::compiler::GraphCompiler::compile(self)?;
            Ok(())
        }

        /// Run the graph's output node on the specified target hardware device.
        pub fn run(&self, device: Device) -> Result<Tensor, Error> {
            let target_node = {
                let inner = self.inner.read().unwrap();
                let mut target_node = None;
                for node in inner.dag.node_indices() {
                    if inner
                        .dag
                        .neighbors_directed(node, petgraph::Direction::Outgoing)
                        .count()
                        == 0
                    {
                        target_node = Some(node);
                        break;
                    }
                }
                target_node
                    .ok_or_else(|| Error::ExecutionError("Graph has no output node".to_string()))?
            };
            let graph_tensor = GraphTensor {
                id: target_node,
                graph: self.clone(),
            };
            graph_tensor.run(device)
        }
    }

    macro_rules! unary_op {
        ($vis:vis $name:ident, $op:ident) => {
            $vis fn $name(&self) -> Self {
                let shape = self.shape();
                let node = Node {
                    op: Op::$op,
                    shape,
                    dtype: Dtype::F32,
                    tensor_id: TensorId::next(),
                };
                let mut inner = self.graph.inner.write().unwrap();
                let out_id = inner.dag.add_node(node);
                inner.dag.add_edge(self.id, out_id, Edge::Unary);
                GraphTensor {
                    id: out_id,
                    graph: self.graph.clone(),
                }
            }
        };
    }

    macro_rules! binary_op {
        ($name:ident, $op:ident, $broadcast_op:ident, $err_prefix:expr) => {
            pub fn $name(&self, other: GraphTensor) -> Self {
                let self_shape = self.shape();
                let other_shape = other.shape();
                if self_shape == other_shape {
                    let node = Node {
                        op: Op::$op,
                        shape: self_shape,
                        dtype: Dtype::F32,
                        tensor_id: TensorId::next(),
                    };
                    let mut inner = self.graph.inner.write().unwrap();
                    let out_id = inner.dag.add_node(node);
                    inner.dag.add_edge(self.id, out_id, Edge::BinaryLhs);
                    inner.dag.add_edge(other.id, out_id, Edge::BinaryRhs);
                    GraphTensor {
                        id: out_id,
                        graph: self.graph.clone(),
                    }
                } else {
                    let out_shape_opt = broadcast_shapes(&self_shape, &other_shape);
                    check_assert!(
                        self,
                        out_shape_opt.is_some(),
                        format!(
                            "{} shapes must be broadcastable: {:?} vs {:?}",
                            $err_prefix,
                            self_shape.dims(),
                            other_shape.dims()
                        )
                    );
                    let out_shape = out_shape_opt.expect("shapes broadcastable (checked above)");
                    let node = Node {
                        op: Op::$broadcast_op {
                            input_shapes: vec![self_shape, other_shape],
                        },
                        shape: out_shape,
                        dtype: Dtype::F32,
                        tensor_id: TensorId::next(),
                    };
                    let mut inner = self.graph.inner.write().unwrap();
                    let out_id = inner.dag.add_node(node);
                    inner.dag.add_edge(self.id, out_id, Edge::BinaryLhs);
                    inner.dag.add_edge(other.id, out_id, Edge::BinaryRhs);
                    GraphTensor {
                        id: out_id,
                        graph: self.graph.clone(),
                    }
                }
            }
        };
    }

    /// Represents a symbolic tensor handle within a computation graph.
    #[derive(Clone)]
    pub struct GraphTensor {
        pub(crate) id: NodeIndex,
        pub(crate) graph: Graph,
    }

    impl GraphTensor {
        /// Get the raw node index of this tensor handle inside the compute DAG.
        pub fn id(&self) -> NodeIndex {
            self.id
        }

        pub(crate) fn record_error_and_dummy(&self, msg: String) -> Self {
            tracing::warn!("{msg}");
            self.graph.set_error(msg);
            let dummy_tensor = Tensor::new(vec![], Shape::new(vec![]));
            let node = Node {
                op: Op::Input(dummy_tensor.clone()),
                shape: Shape::new(vec![]),
                dtype: Dtype::F32,
                tensor_id: dummy_tensor.id(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Get the compute Graph that this tensor belongs to.
        pub fn graph(&self) -> Graph {
            self.graph.clone()
        }

        /// Get the unique TensorId of this tensor inside the graph.
        pub fn tensor_id(&self) -> TensorId {
            self.graph.inner.read().unwrap().dag[self.id].tensor_id
        }

        /// Get the shape of this tensor.
        pub fn shape(&self) -> Shape {
            self.graph.inner.read().unwrap().dag[self.id].shape.clone()
        }

        /// Get the dtype of this tensor.
        pub fn dtype(&self) -> Dtype {
            self.graph.inner.read().unwrap().dag[self.id].dtype
        }

        /// Assign a specific device where this node's computation should execute.
        pub fn to_device(&self, device: crate::Device) -> Self {
            let tid = {
                let inner = self.graph.inner.read().unwrap();
                inner.dag[self.id].tensor_id
            };
            self.graph
                .inner
                .write()
                .unwrap()
                .device_assignments
                .insert(tid, device);
            self.clone()
        }

        /// Run 2D matrix multiplication between this tensor and another.
        pub fn matmul(&self, other: GraphTensor) -> Self {
            let self_shape = self.shape();
            let other_shape = other.shape();
            check_assert!(self, self_shape.ndim() == 2, "MatMul LHS must be 2D");
            check_assert!(self, other_shape.ndim() == 2, "MatMul RHS must be 2D");
            check_assert!(
                self,
                self_shape.dims()[1] == other_shape.dims()[0],
                format!(
                    "MatMul dimensions mismatch: LHS {:?}, RHS {:?}",
                    self_shape.dims(),
                    other_shape.dims()
                )
            );

            let out_shape = Shape::new(vec![self_shape.dims()[0], other_shape.dims()[1]]);
            let node = Node {
                op: Op::MatMul,
                shape: out_shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::BinaryLhs);
            inner.dag.add_edge(other.id, out_id, Edge::BinaryRhs);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        unary_op!(pub relu, Relu);

        binary_op!(add, Add, BroadcastAdd, "Add");

        binary_op!(sub, Sub, BroadcastSub, "Sub");

        binary_op!(mul, Mul, BroadcastMul, "Mul");

        binary_op!(div, Div, BroadcastDiv, "Div");

        unary_op!(pub tanh, Tanh);

        unary_op!(pub sigmoid, Sigmoid);

        unary_op!(pub exp, Exp);

        unary_op!(pub sqrt, Sqrt);

        unary_op!(pub neg, Neg);

        /// Swap dimensions of a 2D matrix.
        pub fn transpose(&self) -> Self {
            let shape = self.shape();
            check_assert!(
                self,
                shape.ndim() == 2,
                "Transpose is only supported for 2D matrices"
            );
            let out_shape = Shape::new(vec![shape.dims()[1], shape.dims()[0]]);
            let node = Node {
                op: Op::Transpose,
                shape: out_shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Reduce the entire tensor to a single-element scalar `Shape([1])` by summing.
        pub fn sum_all(&self) -> Self {
            let node = Node {
                op: Op::SumAll,
                shape: Shape::new(vec![1]),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Reduce along a specific axis by summing, keeping the dimension as 1.
        pub fn sum_dim(&self, axis: usize) -> Self {
            let shape = self.shape();
            check_assert!(self, axis < shape.ndim(), "SumDim axis out of bounds");
            let mut new_dims = shape.dims().to_vec();
            new_dims[axis] = 1;
            let out_shape = Shape::new(new_dims);
            let node = Node {
                op: Op::SumDim { axis },
                shape: out_shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Reshape the tensor to a new shape with the same number of elements.
        pub fn reshape(&self, shape: Shape) -> Self {
            check_assert!(
                self,
                self.shape().num_elements() == shape.num_elements(),
                format!(
                    "Reshape dimensions must have the same number of elements: {} vs {}",
                    self.shape().num_elements(),
                    shape.num_elements()
                )
            );
            let node = Node {
                op: Op::Reshape {
                    shape: shape.clone(),
                },
                shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Compute softmax along the last dimension of the tensor.
        pub fn softmax(&self) -> Self {
            let shape = self.shape();
            let node = Node {
                op: Op::Softmax,
                shape: shape.clone(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Internal backprop op: computes the gradient of softmax.
        fn softmax_grad(&self, grad: GraphTensor) -> Self {
            let shape = self.shape();
            let node = Node {
                op: Op::SoftmaxGrad,
                shape: shape.clone(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::BinaryLhs);
            inner.dag.add_edge(grad.id, out_id, Edge::BinaryRhs);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        unary_op!(step, Step);

        /// Concatenate multiple tensors along an axis.
        pub fn concat(tensors: &[GraphTensor], axis: usize) -> GraphTensor {
            if tensors.is_empty() {
                let graph = Graph::new();
                graph.set_error("Cannot concatenate empty list of tensors".to_string());
                let dummy_tensor = Tensor::new(vec![], Shape::new(vec![]));
                let node = Node {
                    op: Op::Input(dummy_tensor.clone()),
                    shape: Shape::new(vec![]),
                    dtype: Dtype::F32,
                    tensor_id: dummy_tensor.id(),
                };
                let out_id = {
                    let mut inner = graph.inner.write().unwrap();
                    inner.dag.add_node(node)
                };
                return GraphTensor { id: out_id, graph };
            }
            let graph = tensors[0].graph.clone();

            // Calculate shape
            let first_shape = tensors[0].shape();
            let mut concat_dim = 0;
            for t in tensors {
                check_assert!(
                    tensors[0],
                    t.shape().ndim() == first_shape.ndim(),
                    "Tensors must have same number of dimensions to concatenate"
                );
                for d in 0..first_shape.ndim() {
                    if d != axis {
                        check_assert!(
                            tensors[0],
                            t.shape().dims()[d] == first_shape.dims()[d],
                            format!(
                                "Dimension mismatch along axis {}: {} vs {}",
                                d,
                                t.shape().dims()[d],
                                first_shape.dims()[d]
                            )
                        );
                    }
                }
                concat_dim += t.shape().dims()[axis];
            }
            let mut out_dims = first_shape.dims().to_vec();
            out_dims[axis] = concat_dim;
            let out_shape = Shape::new(out_dims);

            let node = Node {
                op: Op::Concat { axis },
                shape: out_shape,
                dtype: tensors[0].dtype(),
                tensor_id: TensorId::next(),
            };

            let id = {
                let mut inner = graph.inner.write().unwrap();
                let id = inner.dag.add_node(node);

                // Add edges from inputs to output
                for (idx, t) in tensors.iter().enumerate() {
                    inner.dag.add_edge(t.id, id, Edge::Index(idx));
                }
                id
            };

            GraphTensor { id, graph }
        }

        /// Apply Layer Normalization.
        pub fn layernorm(
            &self,
            weight: GraphTensor,
            bias: GraphTensor,
            epsilon: f32,
        ) -> GraphTensor {
            let out_shape = self.shape();
            let node = Node {
                op: Op::LayerNorm { epsilon },
                shape: out_shape,
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);

            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(weight.id, id, Edge::Index(1));
            inner.dag.add_edge(bias.id, id, Edge::Index(2));

            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Apply Root Mean Square Normalization (RMSNorm).
        pub fn rmsnorm(&self, weight: GraphTensor, epsilon: f32) -> GraphTensor {
            let out_shape = self.shape();
            let node = Node {
                op: Op::RmsNorm { epsilon },
                shape: out_shape,
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(weight.id, id, Edge::Index(1));
            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Apply 2D Convolution (NCHW format, weight OIHW).
        pub fn conv2d(
            &self,
            weight: GraphTensor,
            bias: Option<GraphTensor>,
            stride: usize,
            padding: usize,
        ) -> GraphTensor {
            let x_shape = self.shape();
            let w_shape = weight.shape();
            check_assert!(self, x_shape.ndim() == 4, "Conv2d input must be 4D (NCHW)");
            check_assert!(self, w_shape.ndim() == 4, "Conv2d weight must be 4D (OIHW)");

            let batch = x_shape.dims()[0];
            let in_channels = x_shape.dims()[1];
            let in_height = x_shape.dims()[2];
            let in_width = x_shape.dims()[3];

            let out_channels = w_shape.dims()[0];
            check_assert!(self, w_shape.dims()[1] == in_channels, format!("Weight input channels must match input channels: weight channels = {}, input channels = {}", w_shape.dims()[1], in_channels));
            let kh = w_shape.dims()[2];
            let kw = w_shape.dims()[3];

            let out_height = (in_height + 2 * padding - kh) / stride + 1;
            let out_width = (in_width + 2 * padding - kw) / stride + 1;

            let out_shape = Shape::new(vec![batch, out_channels, out_height, out_width]);

            let node = Node {
                op: Op::Conv2d { stride, padding },
                shape: out_shape,
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);

            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(weight.id, id, Edge::Index(1));
            if let Some(b) = bias {
                inner.dag.add_edge(b.id, id, Edge::Index(2));
            }

            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Slice a tensor along an axis.
        pub fn slice(&self, axis: usize, start: usize, end: usize) -> GraphTensor {
            let shape = self.shape();
            let mut out_dims = shape.dims().to_vec();
            check_assert!(self, axis < shape.ndim(), "Slice axis out of bounds");
            check_assert!(
                self,
                start <= end && end <= out_dims[axis],
                format!(
                    "Slice indices out of bounds: {}..{} for dimension of size {}",
                    start, end, out_dims[axis]
                )
            );
            out_dims[axis] = end - start;
            let out_shape = Shape::new(out_dims);

            let node = Node {
                op: Op::Slice { axis, start, end },
                shape: out_shape,
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, id, Edge::Unary);

            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Internal backprop op: embeds dy into zero tensor of shape x.
        fn slice_grad(
            &self,
            dy: GraphTensor,
            axis: usize,
            start: usize,
            end: usize,
        ) -> GraphTensor {
            let node = Node {
                op: Op::SliceGrad { axis, start, end },
                shape: self.shape(),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(dy.id, id, Edge::Index(1));
            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Internal backprop op: LayerNorm gradient wrt input x.
        fn layernorm_grad_x(
            &self,
            weight: GraphTensor,
            dy: GraphTensor,
            epsilon: f32,
        ) -> GraphTensor {
            let node = Node {
                op: Op::LayerNormGradX { epsilon },
                shape: self.shape(),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(weight.id, id, Edge::Index(1));
            inner.dag.add_edge(dy.id, id, Edge::Index(2));
            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Internal backprop op: LayerNorm gradient wrt weight.
        fn layernorm_grad_w(&self, dy: GraphTensor, epsilon: f32) -> GraphTensor {
            let shape = self.shape();
            let dims = shape.dims();
            let last_dim = dims[dims.len() - 1];
            let node = Node {
                op: Op::LayerNormGradW { epsilon },
                shape: Shape::new(vec![last_dim]),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(dy.id, id, Edge::Index(1));
            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Internal backprop op: LayerNorm gradient wrt bias.
        fn layernorm_grad_b(dy: GraphTensor) -> GraphTensor {
            let dy_shape = dy.shape();
            let dims = dy_shape.dims();
            let last_dim = dims[dims.len() - 1];
            let node = Node {
                op: Op::LayerNormGradB,
                shape: Shape::new(vec![last_dim]),
                dtype: dy.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = dy.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(dy.id, id, Edge::Index(0));
            GraphTensor {
                id,
                graph: dy.graph.clone(),
            }
        }

        /// Internal backprop op: Conv2d gradient wrt input x.
        fn conv2d_grad_x(
            &self,
            weight: GraphTensor,
            dy: GraphTensor,
            stride: usize,
            padding: usize,
        ) -> GraphTensor {
            let node = Node {
                op: Op::Conv2dGradX { stride, padding },
                shape: self.shape(),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(weight.id, id, Edge::Index(1));
            inner.dag.add_edge(dy.id, id, Edge::Index(2));
            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Internal backprop op: Conv2d gradient wrt weight.
        fn conv2d_grad_w(
            &self,
            weight: GraphTensor,
            dy: GraphTensor,
            stride: usize,
            padding: usize,
        ) -> GraphTensor {
            let node = Node {
                op: Op::Conv2dGradW { stride, padding },
                shape: weight.shape(),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, id, Edge::Index(0));
            inner.dag.add_edge(weight.id, id, Edge::Index(1));
            inner.dag.add_edge(dy.id, id, Edge::Index(2));
            GraphTensor {
                id,
                graph: self.graph.clone(),
            }
        }

        /// Internal backprop op: Conv2d gradient wrt bias.
        fn conv2d_grad_b(dy: GraphTensor) -> GraphTensor {
            let out_channels = dy.shape().dims()[1];
            let node = Node {
                op: Op::Conv2dGradB,
                shape: Shape::new(vec![out_channels]),
                dtype: dy.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = dy.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(dy.id, id, Edge::Index(0));
            GraphTensor {
                id,
                graph: dy.graph.clone(),
            }
        }

        /// Run 3D batched matrix multiplication between this tensor and another.
        pub fn batched_matmul(&self, other: GraphTensor) -> Self {
            let self_shape = self.shape();
            let other_shape = other.shape();
            check_assert!(
                self,
                self_shape.ndim() == 3,
                "BatchedMatMul LHS must be 3D [B, M, K]"
            );
            check_assert!(
                self,
                other_shape.ndim() == 3,
                "BatchedMatMul RHS must be 3D [B, K, N]"
            );
            check_assert!(
                self,
                self_shape.dims()[0] == other_shape.dims()[0],
                "Batch dimensions must match"
            );
            check_assert!(
                self,
                self_shape.dims()[2] == other_shape.dims()[1],
                "Inner dimension K must match"
            );

            let out_shape = Shape::new(vec![
                self_shape.dims()[0],
                self_shape.dims()[1],
                other_shape.dims()[2],
            ]);
            let node = Node {
                op: Op::BatchedMatMul,
                shape: out_shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::BinaryLhs);
            inner.dag.add_edge(other.id, out_id, Edge::BinaryRhs);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Transpose the last two dimensions of a 3D tensor: [B, M, N] -> [B, N, M].
        pub fn batched_transpose(&self) -> Self {
            let shape = self.shape();
            check_assert!(self, shape.ndim() == 3, "BatchedTranspose input must be 3D");
            let out_shape = Shape::new(vec![shape.dims()[0], shape.dims()[2], shape.dims()[1]]);
            let node = Node {
                op: Op::BatchedTranspose,
                shape: out_shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Max pooling 2D.
        pub fn max_pool2d(&self, pool_size: usize, stride: usize, padding: usize) -> Self {
            let shape = self.shape();
            check_assert!(
                self,
                shape.ndim() == 4,
                "MaxPool2d input must be 4D [B, C, H, W]"
            );
            let b = shape.dims()[0];
            let c = shape.dims()[1];
            let h = shape.dims()[2];
            let w = shape.dims()[3];
            let h_out = (h + 2 * padding - pool_size) / stride + 1;
            let w_out = (w + 2 * padding - pool_size) / stride + 1;

            let out_shape = Shape::new(vec![b, c, h_out, w_out]);
            let node = Node {
                op: Op::MaxPool2d {
                    pool_size,
                    stride,
                    padding,
                },
                shape: out_shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Average pooling 2D.
        pub fn avg_pool2d(&self, pool_size: usize, stride: usize, padding: usize) -> Self {
            let shape = self.shape();
            check_assert!(
                self,
                shape.ndim() == 4,
                "AvgPool2d input must be 4D [B, C, H, W]"
            );
            let b = shape.dims()[0];
            let c = shape.dims()[1];
            let h = shape.dims()[2];
            let w = shape.dims()[3];
            let h_out = (h + 2 * padding - pool_size) / stride + 1;
            let w_out = (w + 2 * padding - pool_size) / stride + 1;

            let out_shape = Shape::new(vec![b, c, h_out, w_out]);
            let node = Node {
                op: Op::AvgPool2d {
                    pool_size,
                    stride,
                    padding,
                },
                shape: out_shape,
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Scaled Dot-Product Attention: Softmax(Q * K^T * scale) * V.
        pub fn attention(&self, k: GraphTensor, v: GraphTensor, scale: f32) -> Self {
            let q_shape = self.shape();
            let k_shape = k.shape();
            let v_shape = v.shape();
            check_assert!(
                self,
                q_shape.ndim() == 3,
                "Attention Q must be 3D [B, S, D]"
            );
            check_assert!(
                self,
                k_shape.ndim() == 3,
                "Attention K must be 3D [B, S, D]"
            );
            check_assert!(
                self,
                v_shape.ndim() == 3,
                "Attention V must be 3D [B, S, D]"
            );

            let node = Node {
                op: Op::Attention { scale },
                shape: q_shape.clone(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };

            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Index(0));
            inner.dag.add_edge(k.id, out_id, Edge::Index(1));
            inner.dag.add_edge(v.id, out_id, Edge::Index(2));

            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Cast the tensor to a different dtype.
        pub fn cast(&self, dtype: Dtype) -> Self {
            let current_dtype = self.dtype();
            if current_dtype == dtype {
                return self.clone();
            }
            let (op, target_dtype) = match (current_dtype, dtype) {
                (Dtype::F32, Dtype::F16) => (Op::CastF32ToF16, Dtype::F16),
                (Dtype::F16, Dtype::F32) => (Op::CastF16ToF32, Dtype::F32),
                (Dtype::F32, Dtype::BF16) => (Op::CastF32ToBF16, Dtype::BF16),
                (Dtype::BF16, Dtype::F32) => (Op::CastBF16ToF32, Dtype::F32),
                _ => return self.clone(),
            };
            let shape = self.shape();
            let node = Node {
                op,
                shape: shape.clone(),
                dtype: target_dtype,
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Unary);
            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Causal (masked) Attention: Softmax(Q * K^T * scale + mask) * V.
        pub fn causal_attention(
            &self,
            k: GraphTensor,
            v: GraphTensor,
            scale: f32,
            num_heads: usize,
        ) -> Self {
            let q_shape = self.shape();
            let node = Node {
                op: Op::CausalAttention { scale, num_heads },
                shape: q_shape.clone(),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Index(0));
            inner.dag.add_edge(k.id, out_id, Edge::Index(1));
            inner.dag.add_edge(v.id, out_id, Edge::Index(2));
            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Multi-Head Attention.
        pub fn multi_head_attention(
            &self,
            k: GraphTensor,
            v: GraphTensor,
            scale: f32,
            num_heads: usize,
        ) -> Self {
            let q_shape = self.shape();
            let node = Node {
                op: Op::MultiHeadAttention { scale, num_heads },
                shape: q_shape.clone(),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Index(0));
            inner.dag.add_edge(k.id, out_id, Edge::Index(1));
            inner.dag.add_edge(v.id, out_id, Edge::Index(2));
            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        /// Flash Attention (tiled, O(1) memory w.r.t. sequence length).
        pub fn flash_attention(
            &self,
            k: GraphTensor,
            v: GraphTensor,
            scale: f32,
            causal: bool,
        ) -> Self {
            let q_shape = self.shape();
            let node = Node {
                op: Op::FlashAttention { scale, causal },
                shape: q_shape.clone(),
                dtype: self.dtype(),
                tensor_id: TensorId::next(),
            };
            let mut inner = self.graph.inner.write().unwrap();
            let out_id = inner.dag.add_node(node);
            inner.dag.add_edge(self.id, out_id, Edge::Index(0));
            inner.dag.add_edge(k.id, out_id, Edge::Index(1));
            inner.dag.add_edge(v.id, out_id, Edge::Index(2));
            GraphTensor {
                id: out_id,
                graph: self.graph.clone(),
            }
        }

        fn max_pool2d_grad(
            dy: GraphTensor,
            x: GraphTensor,
            pool_size: usize,
            stride: usize,
            padding: usize,
        ) -> Self {
            let node = Node {
                op: Op::MaxPool2dGrad {
                    pool_size,
                    stride,
                    padding,
                },
                shape: x.shape(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };
            let mut inner = dy.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(dy.id, id, Edge::Index(0));
            inner.dag.add_edge(x.id, id, Edge::Index(1));
            GraphTensor {
                id,
                graph: dy.graph.clone(),
            }
        }

        fn avg_pool2d_grad(
            dy: GraphTensor,
            x: GraphTensor,
            pool_size: usize,
            stride: usize,
            padding: usize,
        ) -> Self {
            let node = Node {
                op: Op::AvgPool2dGrad {
                    pool_size,
                    stride,
                    padding,
                },
                shape: x.shape(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };
            let mut inner = dy.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(dy.id, id, Edge::Index(0));
            inner.dag.add_edge(x.id, id, Edge::Index(1));
            GraphTensor {
                id,
                graph: dy.graph.clone(),
            }
        }

        fn attention_grad_q(
            dy: GraphTensor,
            q: GraphTensor,
            k: GraphTensor,
            v: GraphTensor,
            scale: f32,
        ) -> Self {
            let node = Node {
                op: Op::AttentionGradQ { scale },
                shape: q.shape(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };
            let mut inner = dy.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(dy.id, id, Edge::Index(0));
            inner.dag.add_edge(q.id, id, Edge::Index(1));
            inner.dag.add_edge(k.id, id, Edge::Index(2));
            inner.dag.add_edge(v.id, id, Edge::Index(3));
            GraphTensor {
                id,
                graph: dy.graph.clone(),
            }
        }

        fn attention_grad_k(
            dy: GraphTensor,
            q: GraphTensor,
            k: GraphTensor,
            v: GraphTensor,
            scale: f32,
        ) -> Self {
            let node = Node {
                op: Op::AttentionGradK { scale },
                shape: k.shape(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };
            let mut inner = dy.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(dy.id, id, Edge::Index(0));
            inner.dag.add_edge(q.id, id, Edge::Index(1));
            inner.dag.add_edge(k.id, id, Edge::Index(2));
            inner.dag.add_edge(v.id, id, Edge::Index(3));
            GraphTensor {
                id,
                graph: dy.graph.clone(),
            }
        }

        fn attention_grad_v(
            dy: GraphTensor,
            q: GraphTensor,
            k: GraphTensor,
            v: GraphTensor,
            scale: f32,
        ) -> Self {
            let node = Node {
                op: Op::AttentionGradV { scale },
                shape: v.shape(),
                dtype: Dtype::F32,
                tensor_id: TensorId::next(),
            };
            let mut inner = dy.graph.inner.write().unwrap();
            let id = inner.dag.add_node(node);
            inner.dag.add_edge(dy.id, id, Edge::Index(0));
            inner.dag.add_edge(q.id, id, Edge::Index(1));
            inner.dag.add_edge(k.id, id, Edge::Index(2));
            inner.dag.add_edge(v.id, id, Edge::Index(3));
            GraphTensor {
                id,
                graph: dy.graph.clone(),
            }
        }

        /// Compute gradients of this scalar tensor with respect to all other nodes in the graph
        /// using reverse-mode automatic differentiation. Returns a map from node indices to gradient handles.
        pub fn backward(&self) -> Result<std::collections::HashMap<NodeIndex, GraphTensor>, Error> {
            if let Some(err_msg) = self.graph.get_error() {
                return Err(Error::ExecutionError(err_msg));
            }
            let shape = self.shape();
            if shape.num_elements() != 1 {
                return Err(Error::ExecutionError(
                    "backward() can only be called on scalar tensors (shape [1] or 1 element)"
                        .to_string(),
                ));
            }

            // 1. Perform topological sort of nodes reachable from this node
            let mut visited = std::collections::HashSet::new();
            let mut order = Vec::new();

            fn dfs(
                node: NodeIndex,
                dag: &DiGraph<Node, Edge>,
                visited: &mut std::collections::HashSet<NodeIndex>,
                order: &mut Vec<NodeIndex>,
            ) {
                if !visited.insert(node) {
                    return;
                }
                use petgraph::visit::EdgeRef;
                for edge in dag.edges_directed(node, petgraph::Direction::Incoming) {
                    dfs(edge.source(), dag, visited, order);
                }
                order.push(node);
            }

            {
                let inner = self.graph.inner.read().unwrap();
                dfs(self.id, &inner.dag, &mut visited, &mut order);
            }

            // 2. Initialize the gradient map
            let mut grads = std::collections::HashMap::new();
            let one_data = vec![1.0];
            let one_tensor = self.graph.tensor(one_data, Shape::new(vec![1]));
            grads.insert(self.id, one_tensor);

            // 3. Propagate gradients in reverse topological order (from outputs to inputs)
            for &node_idx in order.iter().rev() {
                let d_out = match grads.get(&node_idx) {
                    Some(g) => g.clone(),
                    None => continue,
                };

                let node_op = {
                    let inner = self.graph.inner.read().unwrap();
                    inner.dag[node_idx].op.clone()
                };

                match &node_op {
                    Op::Input(_) => {}
                    Op::MatMul => {
                        let (lhs_node, rhs_node) = {
                            let inner = self.graph.inner.read().unwrap();
                            get_binary_inputs(&inner.dag, node_idx)?
                        };

                        let lhs = GraphTensor {
                            id: lhs_node,
                            graph: self.graph.clone(),
                        };
                        let rhs = GraphTensor {
                            id: rhs_node,
                            graph: self.graph.clone(),
                        };

                        let rhs_t = rhs.transpose();
                        let d_lhs = d_out.matmul(rhs_t);

                        let lhs_t = lhs.transpose();
                        let d_rhs = lhs_t.matmul(d_out);

                        accumulate_grad(&mut grads, lhs_node, d_lhs);
                        accumulate_grad(&mut grads, rhs_node, d_rhs);
                    }
                    Op::Relu => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };
                        let dx = d_out.mul(x.step());
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Add | Op::BroadcastAdd { .. } => {
                        let (lhs_node, rhs_node) = {
                            let inner = self.graph.inner.read().unwrap();
                            get_binary_inputs(&inner.dag, node_idx)?
                        };
                        let (lhs_shape, rhs_shape) = {
                            let inner = self.graph.inner.read().unwrap();
                            (
                                inner.dag[lhs_node].shape.clone(),
                                inner.dag[rhs_node].shape.clone(),
                            )
                        };
                        let d_lhs = reduce_grad_if_broadcast(d_out.clone(), &lhs_shape);
                        let d_rhs = reduce_grad_if_broadcast(d_out, &rhs_shape);

                        accumulate_grad(&mut grads, lhs_node, d_lhs);
                        accumulate_grad(&mut grads, rhs_node, d_rhs);
                    }
                    Op::Sub | Op::BroadcastSub { .. } => {
                        let (lhs_node, rhs_node) = {
                            let inner = self.graph.inner.read().unwrap();
                            get_binary_inputs(&inner.dag, node_idx)?
                        };
                        let (lhs_shape, rhs_shape) = {
                            let inner = self.graph.inner.read().unwrap();
                            (
                                inner.dag[lhs_node].shape.clone(),
                                inner.dag[rhs_node].shape.clone(),
                            )
                        };
                        let d_lhs = reduce_grad_if_broadcast(d_out.clone(), &lhs_shape);
                        let d_rhs = reduce_grad_if_broadcast(d_out.neg(), &rhs_shape);

                        accumulate_grad(&mut grads, lhs_node, d_lhs);
                        accumulate_grad(&mut grads, rhs_node, d_rhs);
                    }
                    Op::Mul | Op::BroadcastMul { .. } => {
                        let (lhs_node, rhs_node) = {
                            let inner = self.graph.inner.read().unwrap();
                            get_binary_inputs(&inner.dag, node_idx)?
                        };
                        let lhs = GraphTensor {
                            id: lhs_node,
                            graph: self.graph.clone(),
                        };
                        let rhs = GraphTensor {
                            id: rhs_node,
                            graph: self.graph.clone(),
                        };

                        let (lhs_shape, rhs_shape) = {
                            let inner = self.graph.inner.read().unwrap();
                            (
                                inner.dag[lhs_node].shape.clone(),
                                inner.dag[rhs_node].shape.clone(),
                            )
                        };
                        let d_lhs = reduce_grad_if_broadcast(d_out.mul(rhs), &lhs_shape);
                        let d_rhs = reduce_grad_if_broadcast(d_out.mul(lhs), &rhs_shape);

                        accumulate_grad(&mut grads, lhs_node, d_lhs);
                        accumulate_grad(&mut grads, rhs_node, d_rhs);
                    }
                    Op::Div | Op::BroadcastDiv { .. } => {
                        let (lhs_node, rhs_node) = {
                            let inner = self.graph.inner.read().unwrap();
                            get_binary_inputs(&inner.dag, node_idx)?
                        };
                        let lhs = GraphTensor {
                            id: lhs_node,
                            graph: self.graph.clone(),
                        };
                        let rhs = GraphTensor {
                            id: rhs_node,
                            graph: self.graph.clone(),
                        };

                        let (lhs_shape, rhs_shape) = {
                            let inner = self.graph.inner.read().unwrap();
                            (
                                inner.dag[lhs_node].shape.clone(),
                                inner.dag[rhs_node].shape.clone(),
                            )
                        };
                        let d_lhs = reduce_grad_if_broadcast(d_out.div(rhs.clone()), &lhs_shape);
                        let d_rhs = reduce_grad_if_broadcast(
                            d_out.neg().mul(lhs).div(rhs.clone().mul(rhs)),
                            &rhs_shape,
                        );

                        accumulate_grad(&mut grads, lhs_node, d_lhs);
                        accumulate_grad(&mut grads, rhs_node, d_rhs);
                    }
                    Op::Tanh => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };

                        let t = x.tanh();
                        let one = self
                            .graph
                            .tensor(vec![1.0; t.shape().num_elements()], t.shape());
                        let dx = d_out.mul(one.sub(t.clone().mul(t)));
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Sigmoid => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };

                        let s = x.sigmoid();
                        let one = self
                            .graph
                            .tensor(vec![1.0; s.shape().num_elements()], s.shape());
                        let dx = d_out.mul(s.clone().mul(one.sub(s)));
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Exp => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };

                        let dx = d_out.mul(x.exp());
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Sqrt => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };

                        let half = self
                            .graph
                            .tensor(vec![0.5; x.shape().num_elements()], x.shape());
                        let dx = d_out.mul(half).div(x.sqrt());
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Neg => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let dx = d_out.neg();
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Transpose => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let dx = d_out.transpose();
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::SumAll => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let in_shape = self.graph.inner.read().unwrap().dag[input_node]
                            .shape
                            .clone();
                        let ones = self
                            .graph
                            .tensor(vec![1.0; in_shape.num_elements()], in_shape);
                        let dx = ones.mul(d_out);
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::SumDim { .. } => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let in_shape = self.graph.inner.read().unwrap().dag[input_node]
                            .shape
                            .clone();
                        let ones = self
                            .graph
                            .tensor(vec![1.0; in_shape.num_elements()], in_shape);
                        let dx = ones.mul(d_out);
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Reshape { .. } => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let in_shape = self.graph.inner.read().unwrap().dag[input_node]
                            .shape
                            .clone();
                        let dx = d_out.reshape(in_shape);
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Softmax => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let softmax_out = GraphTensor {
                            id: node_idx,
                            graph: self.graph.clone(),
                        };
                        let dx = softmax_out.softmax_grad(d_out);
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Concat { axis } => {
                        let inputs = {
                            let inner = self.graph.inner.read().unwrap();
                            get_indexed_inputs(&inner.dag, node_idx)?
                        };
                        let mut start = 0;
                        for &input_node in &inputs {
                            let input_shape = {
                                let inner = self.graph.inner.read().unwrap();
                                inner.dag[input_node].shape.clone()
                            };
                            let size = input_shape.dims()[*axis];
                            let dx = d_out.slice(*axis, start, start + size);
                            accumulate_grad(&mut grads, input_node, dx);
                            start += size;
                        }
                    }
                    Op::LayerNorm { epsilon } => {
                        let inputs = {
                            let inner = self.graph.inner.read().unwrap();
                            get_indexed_inputs(&inner.dag, node_idx)?
                        };
                        assert_eq!(inputs.len(), 3);
                        let x_node = inputs[0];
                        let w_node = inputs[1];
                        let b_node = inputs[2];

                        let x = GraphTensor {
                            id: x_node,
                            graph: self.graph.clone(),
                        };
                        let w = GraphTensor {
                            id: w_node,
                            graph: self.graph.clone(),
                        };

                        let dx = x.layernorm_grad_x(w, d_out.clone(), *epsilon);
                        let dw = x.layernorm_grad_w(d_out.clone(), *epsilon);
                        let db = GraphTensor::layernorm_grad_b(d_out);

                        accumulate_grad(&mut grads, x_node, dx);
                        accumulate_grad(&mut grads, w_node, dw);
                        accumulate_grad(&mut grads, b_node, db);
                    }
                    Op::Conv2d { stride, padding } => {
                        let inputs = {
                            let inner = self.graph.inner.read().unwrap();
                            get_indexed_inputs(&inner.dag, node_idx)?
                        };
                        let x_node = inputs[0];
                        let w_node = inputs[1];

                        let x = GraphTensor {
                            id: x_node,
                            graph: self.graph.clone(),
                        };
                        let w = GraphTensor {
                            id: w_node,
                            graph: self.graph.clone(),
                        };

                        let dx = x.conv2d_grad_x(w.clone(), d_out.clone(), *stride, *padding);
                        let dw = x.conv2d_grad_w(w, d_out.clone(), *stride, *padding);

                        accumulate_grad(&mut grads, x_node, dx);
                        accumulate_grad(&mut grads, w_node, dw);

                        if inputs.len() == 3 {
                            let b_node = inputs[2];
                            let db = GraphTensor::conv2d_grad_b(d_out);
                            accumulate_grad(&mut grads, b_node, db);
                        }
                    }
                    Op::Slice { axis, start, end } => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };
                        let dx = x.slice_grad(d_out, *axis, *start, *end);
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::BatchedMatMul => {
                        let (lhs_node, rhs_node) = {
                            let inner = self.graph.inner.read().unwrap();
                            get_binary_inputs(&inner.dag, node_idx)?
                        };
                        let lhs = GraphTensor {
                            id: lhs_node,
                            graph: self.graph.clone(),
                        };
                        let rhs = GraphTensor {
                            id: rhs_node,
                            graph: self.graph.clone(),
                        };

                        let d_lhs = d_out.batched_matmul(rhs.batched_transpose());
                        let d_rhs = lhs.batched_transpose().batched_matmul(d_out);

                        accumulate_grad(&mut grads, lhs_node, d_lhs);
                        accumulate_grad(&mut grads, rhs_node, d_rhs);
                    }
                    Op::BatchedTranspose => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let dx = d_out.batched_transpose();
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::MaxPool2d {
                        pool_size,
                        stride,
                        padding,
                    } => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };
                        let dx =
                            GraphTensor::max_pool2d_grad(d_out, x, *pool_size, *stride, *padding);
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::AvgPool2d {
                        pool_size,
                        stride,
                        padding,
                    } => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        let x = GraphTensor {
                            id: input_node,
                            graph: self.graph.clone(),
                        };
                        let dx =
                            GraphTensor::avg_pool2d_grad(d_out, x, *pool_size, *stride, *padding);
                        accumulate_grad(&mut grads, input_node, dx);
                    }
                    Op::Attention { scale } => {
                        let inputs = {
                            let inner = self.graph.inner.read().unwrap();
                            get_indexed_inputs(&inner.dag, node_idx)?
                        };
                        assert_eq!(inputs.len(), 3);
                        let q_node = inputs[0];
                        let k_node = inputs[1];
                        let v_node = inputs[2];

                        let q = GraphTensor {
                            id: q_node,
                            graph: self.graph.clone(),
                        };
                        let k = GraphTensor {
                            id: k_node,
                            graph: self.graph.clone(),
                        };
                        let v = GraphTensor {
                            id: v_node,
                            graph: self.graph.clone(),
                        };

                        let dq = GraphTensor::attention_grad_q(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dk = GraphTensor::attention_grad_k(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dv = GraphTensor::attention_grad_v(d_out, q, k, v, *scale);

                        accumulate_grad(&mut grads, q_node, dq);
                        accumulate_grad(&mut grads, k_node, dk);
                        accumulate_grad(&mut grads, v_node, dv);
                    }
                    Op::CastF32ToF16 | Op::CastF16ToF32 | Op::CastF32ToBF16 | Op::CastBF16ToF32 => {
                        let input_node = {
                            let inner = self.graph.inner.read().unwrap();
                            get_unary_input(&inner.dag, node_idx)?
                        };
                        accumulate_grad(&mut grads, input_node, d_out);
                    }
                    Op::CausalAttention {
                        scale,
                        num_heads: _num_heads,
                    } => {
                        let inputs = {
                            let inner = self.graph.inner.read().unwrap();
                            get_indexed_inputs(&inner.dag, node_idx)?
                        };
                        let q_node = inputs[0];
                        let k_node = inputs[1];
                        let v_node = inputs[2];
                        let q = GraphTensor {
                            id: q_node,
                            graph: self.graph.clone(),
                        };
                        let k = GraphTensor {
                            id: k_node,
                            graph: self.graph.clone(),
                        };
                        let v = GraphTensor {
                            id: v_node,
                            graph: self.graph.clone(),
                        };
                        let dq = GraphTensor::attention_grad_q(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dk = GraphTensor::attention_grad_k(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dv = GraphTensor::attention_grad_v(d_out, q, k, v, *scale);
                        accumulate_grad(&mut grads, q_node, dq);
                        accumulate_grad(&mut grads, k_node, dk);
                        accumulate_grad(&mut grads, v_node, dv);
                    }
                    Op::MultiHeadAttention {
                        scale,
                        num_heads: _num_heads,
                    } => {
                        let inputs = {
                            let inner = self.graph.inner.read().unwrap();
                            get_indexed_inputs(&inner.dag, node_idx)?
                        };
                        let q_node = inputs[0];
                        let k_node = inputs[1];
                        let v_node = inputs[2];
                        let q = GraphTensor {
                            id: q_node,
                            graph: self.graph.clone(),
                        };
                        let k = GraphTensor {
                            id: k_node,
                            graph: self.graph.clone(),
                        };
                        let v = GraphTensor {
                            id: v_node,
                            graph: self.graph.clone(),
                        };
                        let dq = GraphTensor::attention_grad_q(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dk = GraphTensor::attention_grad_k(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dv = GraphTensor::attention_grad_v(d_out, q, k, v, *scale);
                        accumulate_grad(&mut grads, q_node, dq);
                        accumulate_grad(&mut grads, k_node, dk);
                        accumulate_grad(&mut grads, v_node, dv);
                    }
                    Op::FlashAttention {
                        scale,
                        causal: _causal,
                    } => {
                        let inputs = {
                            let inner = self.graph.inner.read().unwrap();
                            get_indexed_inputs(&inner.dag, node_idx)?
                        };
                        let q_node = inputs[0];
                        let k_node = inputs[1];
                        let v_node = inputs[2];
                        let q = GraphTensor {
                            id: q_node,
                            graph: self.graph.clone(),
                        };
                        let k = GraphTensor {
                            id: k_node,
                            graph: self.graph.clone(),
                        };
                        let v = GraphTensor {
                            id: v_node,
                            graph: self.graph.clone(),
                        };
                        let dq = GraphTensor::attention_grad_q(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dk = GraphTensor::attention_grad_k(
                            d_out.clone(),
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            *scale,
                        );
                        let dv = GraphTensor::attention_grad_v(d_out, q, k, v, *scale);
                        accumulate_grad(&mut grads, q_node, dq);
                        accumulate_grad(&mut grads, k_node, dk);
                        accumulate_grad(&mut grads, v_node, dv);
                    }
                    Op::SoftmaxGrad
                    | Op::Step
                    | Op::SliceGrad { .. }
                    | Op::LayerNormGradX { .. }
                    | Op::LayerNormGradW { .. }
                    | Op::LayerNormGradB
                    | Op::Conv2dGradX { .. }
                    | Op::Conv2dGradW { .. }
                    | Op::Conv2dGradB
                    | Op::MaxPool2dGrad { .. }
                    | Op::AvgPool2dGrad { .. }
                    | Op::AttentionGradQ { .. }
                    | Op::AttentionGradK { .. }
                    | Op::AttentionGradV { .. }
                    | Op::RmsNorm { .. } => {}
                }
            }

            Ok(grads)
        }

        /// Execute this tensor's graph and return the evaluated Tensor.
        /// If the graph's `compile_on_run` flag is set (see [`Graph::set_compile_on_run`]),
        /// the compiler optimisation passes run first.
        pub fn run(&self, device: Device) -> Result<Tensor, Error> {
            let should_compile = {
                let inner = self.graph.inner.read().unwrap();
                inner.compile_on_run
            };
            if should_compile {
                self.graph.compile()?;
            }
            crate::runtime::execute(self, device)
        }

        /// Execute the graph's computation without reading back the result.
        /// All compute kernels are submitted to the device, but the output
        /// is left in GPU memory. Returns `Ok(())` after submission.
        /// Use [`run`] on the same graph to read back results.
        /// If the graph's `compile_on_run` flag is set, the compiler optimisation
        /// passes run first.
        pub fn run_no_readback(&self, device: Device) -> Result<(), Error> {
            let should_compile = {
                let inner = self.graph.inner.read().unwrap();
                inner.compile_on_run
            };
            if should_compile {
                self.graph.compile()?;
            }
            crate::runtime::execute_no_readback(self, device)
        }
    }

    fn accumulate_grad(
        grads: &mut std::collections::HashMap<NodeIndex, GraphTensor>,
        node: NodeIndex,
        grad: GraphTensor,
    ) {
        if let Some(existing) = grads.get(&node) {
            let new_grad = existing.add(grad);
            grads.insert(node, new_grad);
        } else {
            grads.insert(node, grad);
        }
    }

    fn reduce_grad_if_broadcast(mut grad: GraphTensor, target_shape: &Shape) -> GraphTensor {
        let grad_shape = grad.shape();
        if &grad_shape == target_shape {
            return grad;
        }
        let grad_dims = grad_shape.dims();
        let target_dims = target_shape.dims();
        let diff = grad_dims.len().saturating_sub(target_dims.len());

        for i in 0..grad_dims.len() {
            let target_dim = if i < diff { 1 } else { target_dims[i - diff] };
            if target_dim == 1 && grad_dims[i] > 1 {
                grad = grad.sum_dim(i);
            }
        }

        if grad.shape() != *target_shape {
            grad = grad.reshape(target_shape.clone());
        }
        grad
    }

    pub(crate) fn get_unary_input(
        dag: &DiGraph<Node, Edge>,
        node_id: NodeIndex,
    ) -> Result<NodeIndex, Error> {
        use petgraph::visit::EdgeRef;
        for edge in dag.edges_directed(node_id, petgraph::Direction::Incoming) {
            if edge.weight() == &Edge::Unary {
                return Ok(edge.source());
            }
        }
        Err(Error::ExecutionError(format!(
            "Missing unary input for node {:?}",
            node_id
        )))
    }

    pub(crate) fn get_binary_inputs(
        dag: &DiGraph<Node, Edge>,
        node_id: NodeIndex,
    ) -> Result<(NodeIndex, NodeIndex), Error> {
        use petgraph::visit::EdgeRef;
        let mut lhs = None;
        let mut rhs = None;
        for edge in dag.edges_directed(node_id, petgraph::Direction::Incoming) {
            match edge.weight() {
                Edge::BinaryLhs => lhs = Some(edge.source()),
                Edge::BinaryRhs => rhs = Some(edge.source()),
                _ => {}
            }
        }
        match (lhs, rhs) {
            (Some(l), Some(r)) => Ok((l, r)),
            _ => Err(Error::ExecutionError(format!(
                "Missing binary inputs for node {:?}",
                node_id
            ))),
        }
    }

    pub fn get_indexed_inputs(
        dag: &DiGraph<Node, Edge>,
        node_id: NodeIndex,
    ) -> Result<Vec<NodeIndex>, Error> {
        use petgraph::visit::EdgeRef;
        let mut edges: Vec<_> = dag
            .edges_directed(node_id, petgraph::Direction::Incoming)
            .collect();
        edges.sort_by_key(|e| match e.weight() {
            Edge::BinaryLhs => 0,
            Edge::BinaryRhs => 1,
            Edge::Unary => 0,
            Edge::Index(idx) => *idx,
        });
        Ok(edges.into_iter().map(|e| e.source()).collect())
    }

    /// Broadcast helper to check compatibility and compute resulting shape of two shapes.
    pub fn broadcast_shapes(a: &Shape, b: &Shape) -> Option<Shape> {
        let a_dims = a.dims();
        let b_dims = b.dims();
        let mut out_dims = Vec::new();
        let len = a_dims.len().max(b_dims.len());
        for i in 0..len {
            let dim_a = if i < a_dims.len() {
                a_dims[a_dims.len() - 1 - i]
            } else {
                1
            };
            let dim_b = if i < b_dims.len() {
                b_dims[b_dims.len() - 1 - i]
            } else {
                1
            };
            if dim_a == dim_b {
                out_dims.push(dim_a);
            } else if dim_a == 1 {
                out_dims.push(dim_b);
            } else if dim_b == 1 {
                out_dims.push(dim_a);
            } else {
                return None;
            }
        }
        out_dims.reverse();
        Some(Shape::new(out_dims))
    }
}
pub use graph_mod::{broadcast_shapes, get_indexed_inputs, Edge, Graph, GraphTensor, Node, Op};
