use crate::{Device, Error, GraphTensor, TensorId};
use std::collections::HashMap;

/// Gradient scaler for mixed precision training (fp16).
/// Scales loss before backward pass to prevent underflow,
/// then unscales gradients before optimizer step.
pub struct GradScaler {
    scale: f32,
    scale_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    growth_steps: usize,
    finite_count: usize,
}

impl GradScaler {
    /// Create a new GradScaler.
    pub fn new(
        initial_scale: f32,
        scale_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    ) -> Self {
        Self {
            scale: initial_scale,
            scale_factor,
            backoff_factor,
            growth_interval,
            growth_steps: 0,
            finite_count: 0,
        }
    }

    /// Create a new GradScaler with default parameters.
    pub fn default_config() -> Self {
        Self::new(65536.0, 2.0, 0.5, 2000)
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new(65536.0, 2.0, 0.5, 2000)
    }
}

impl GradScaler {
    /// Returns the current scale factor.
    pub fn current_scale(&self) -> f32 {
        self.scale
    }

    /// Set the current scale factor.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Scale the loss by the current scale factor.
    /// Returns a new GraphTensor with the scaled loss.
    pub fn scale_loss(&self, loss: GraphTensor) -> GraphTensor {
        let scale_tensor = {
            let graph = loss.graph();
            graph.tensor(vec![self.scale], crate::Shape::new(vec![1]))
        };
        loss.mul(scale_tensor)
    }

    /// Unscale gradients by dividing by the scale factor.
    /// Returns a map of TensorId -> unscaled gradient.
    pub fn unscale(
        &self,
        grads: &HashMap<TensorId, GraphTensor>,
        device: Device,
    ) -> Result<HashMap<TensorId, GraphTensor>, Error> {
        let mut unscaled = HashMap::new();
        for (tid, grad) in grads {
            let scale_tensor = {
                let graph = grad.graph();
                graph.tensor(vec![self.scale], crate::Shape::new(vec![1]))
            };
            let unscaled_grad = grad.div(scale_tensor);
            let _ = device; // evaluated lazily via the graph
            unscaled.insert(*tid, unscaled_grad);
        }
        Ok(unscaled)
    }

    /// Check whether gradients are finite (no NaN/Inf values).
    /// Returns false if any gradient contains NaN or Inf.
    pub fn check_grads_finite(
        grads: &HashMap<TensorId, GraphTensor>,
        device: Device,
    ) -> Result<bool, Error> {
        for grad in grads.values() {
            let evaluated = grad.run(device)?;
            let data = evaluated.data();
            if data.iter().any(|&x| x.is_nan() || x.is_infinite()) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Update the scale factor based on whether gradients were finite.
    /// If grads_finite is true and we've accumulated enough growth steps, increase scale.
    /// If grads_finite is false, decrease scale.
    pub fn update(&mut self, grads_finite: bool) {
        if grads_finite {
            self.growth_steps += 1;
            self.finite_count += 1;
            if self.growth_steps >= self.growth_interval {
                self.scale *= self.scale_factor;
                self.growth_steps = 0;
            }
        } else {
            self.scale *= self.backoff_factor;
            self.growth_steps = 0;
            self.finite_count = 0;
        }
    }

    /// Perform a full step: scale loss, backward, check grads, unscale, update.
    pub fn step(
        &mut self,
        loss: GraphTensor,
        optimizer: &mut dyn FnMut(HashMap<TensorId, GraphTensor>, Device) -> Result<(), Error>,
        device: Device,
    ) -> Result<(), Error> {
        let scaled_loss = self.scale_loss(loss);
        let node_grads = scaled_loss.backward()?;
        let graph_ptr = scaled_loss.graph();
        let inner = graph_ptr.inner.read().map_err(|e| {
            Error::ExecutionError(format!("graph lock poisoned in step: {e}"))
        })?;
        let grads: HashMap<TensorId, GraphTensor> = node_grads
            .into_iter()
            .map(|(idx, gt)| {
                let tid = inner
                    .dag
                    .node_weight(idx)
                    .map(|n| n.tensor_id)
                    .unwrap_or_else(TensorId::next);
                (tid, gt)
            })
            .collect();
        drop(inner);
        let finite = Self::check_grads_finite(&grads, device)?;
        if finite {
            let unscaled = self.unscale(&grads, device)?;
            optimizer(unscaled, device)?;
        }
        self.update(finite);
        Ok(())
    }
}
