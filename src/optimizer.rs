use crate::{Device, Error, GraphTensor, TensorId};
use std::collections::HashMap;

// ─────────────────────────────────────────────
//  SGD
// ─────────────────────────────────────────────

/// Stochastic Gradient Descent optimizer with optional momentum and weight decay.
pub struct Sgd {
    params: Vec<GraphTensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: HashMap<TensorId, Vec<f32>>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    pub fn new(params: Vec<GraphTensor>, lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            momentum,
            weight_decay,
            velocities: HashMap::new(),
        }
    }

    /// Perform a single optimization step.
    pub fn step(
        &mut self,
        grads: &HashMap<TensorId, GraphTensor>,
        device: Device,
    ) -> Result<(), Error> {
        for param in &self.params {
            let tid = param.tensor_id();
            if let Some(grad_tensor) = grads.get(&tid) {
                let grad_eval = grad_tensor.run(device)?;
                let grad_data = grad_eval.data();
                let mut param_data = param.run(device)?.data().to_vec();
                assert_eq!(
                    param_data.len(),
                    grad_data.len(),
                    "Parameter and gradient size mismatch"
                );

                let velocity = self
                    .velocities
                    .entry(tid)
                    .or_insert_with(|| vec![0.0; param_data.len()]);

                for i in 0..param_data.len() {
                    let mut g = grad_data[i];
                    if self.weight_decay != 0.0 {
                        g += self.weight_decay * param_data[i];
                    }
                    if self.momentum != 0.0 {
                        velocity[i] = self.momentum * velocity[i] + g;
                        param_data[i] -= self.lr * velocity[i];
                    } else {
                        param_data[i] -= self.lr * g;
                    }
                }
                param.graph().update_input(param.id(), param_data);
            }
        }
        Ok(())
    }

    /// Return the velocity buffer for a parameter, if it exists.
    pub fn get_velocity(&self, tid: TensorId) -> Option<&Vec<f32>> {
        self.velocities.get(&tid)
    }

    /// Overwrite the velocity buffer for a parameter (used when restoring checkpoints).
    pub fn set_velocity(&mut self, tid: TensorId, velocity: Vec<f32>) {
        self.velocities.insert(tid, velocity);
    }
}

// ─────────────────────────────────────────────
//  AdamW
// ─────────────────────────────────────────────

/// AdamW optimizer with decoupled weight decay.
///
/// Implements the algorithm from "Decoupled Weight Decay Regularization"
/// (Loshchilov & Hutter, 2019).
pub struct AdamW {
    params: Vec<GraphTensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step_count: usize,
    /// Per-parameter first moment (mean of gradients).
    m: HashMap<TensorId, Vec<f32>>,
    /// Per-parameter second moment (uncentred variance of gradients).
    v: HashMap<TensorId, Vec<f32>>,
}

impl AdamW {
    /// Create a new AdamW optimizer.
    pub fn new(
        params: Vec<GraphTensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Perform a single optimization step.
    pub fn step(
        &mut self,
        grads: &HashMap<TensorId, GraphTensor>,
        device: Device,
    ) -> Result<(), Error> {
        self.step_count += 1;
        let t = self.step_count as f32;

        let correction1 = 1.0 - self.beta1.powf(t);
        let correction2 = 1.0 - self.beta2.powf(t);

        for param in &self.params {
            let tid = param.tensor_id();
            if let Some(grad_tensor) = grads.get(&tid) {
                let grad_eval = grad_tensor.run(device)?;
                let grad_data = grad_eval.data();
                let mut param_data = param.run(device)?.data().to_vec();
                assert_eq!(
                    param_data.len(),
                    grad_data.len(),
                    "Parameter and gradient size mismatch"
                );

                let m_state = self
                    .m
                    .entry(tid)
                    .or_insert_with(|| vec![0.0; param_data.len()]);
                let v_state = self
                    .v
                    .entry(tid)
                    .or_insert_with(|| vec![0.0; param_data.len()]);

                for i in 0..param_data.len() {
                    let g = grad_data[i];

                    // 1. Decoupled weight decay
                    if self.weight_decay != 0.0 {
                        param_data[i] -= self.lr * self.weight_decay * param_data[i];
                    }

                    // 2. Update first moment
                    m_state[i] = self.beta1 * m_state[i] + (1.0 - self.beta1) * g;

                    // 3. Update second moment
                    v_state[i] = self.beta2 * v_state[i] + (1.0 - self.beta2) * g * g;

                    // 4. Bias-corrected moments
                    let m_hat = m_state[i] / correction1;
                    let v_hat = v_state[i] / correction2;

                    // 5. Parameter update
                    param_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }

                param.graph().update_input(param.id(), param_data);
            }
        }
        Ok(())
    }

    // ── State accessors (used for checkpointing) ──────────────────────────────

    /// Return the first-moment buffer for a parameter.
    pub fn get_m(&self, tid: TensorId) -> Option<&Vec<f32>> {
        self.m.get(&tid)
    }

    /// Return the second-moment buffer for a parameter.
    pub fn get_v(&self, tid: TensorId) -> Option<&Vec<f32>> {
        self.v.get(&tid)
    }

    /// Overwrite both moment buffers for a parameter (used when restoring from a checkpoint).
    pub fn set_moments(&mut self, tid: TensorId, m: Vec<f32>, v: Vec<f32>) {
        self.m.insert(tid, m);
        self.v.insert(tid, v);
    }

    /// Return the total number of optimizer steps taken so far.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Restore the step counter from a checkpoint.
    pub fn set_step_count(&mut self, step_count: usize) {
        self.step_count = step_count;
    }
}
