use crate::tensor::{Dtype, Shape};
use sha2::{Digest, Sha256};

/// Abstract Syntax Tree (AST) representing elementwise computations.
/// Used for dynamically compiling fused WGSL shaders and CPU-side execution.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Reference to an input buffer by its position.
    Input(usize),
    /// A constant scalar f32 value.
    Scalar(f32),
    /// Addition of two sub-expressions.
    Add(Box<Expr>, Box<Expr>),
    /// Multiplication of two sub-expressions.
    Mul(Box<Expr>, Box<Expr>),
    /// Subtraction of two sub-expressions.
    Sub(Box<Expr>, Box<Expr>),
    /// Division of two sub-expressions.
    Div(Box<Expr>, Box<Expr>),
    /// Rectified Linear Unit activation function.
    Relu(Box<Expr>),
    /// Hyperbolic tangent function.
    Tanh(Box<Expr>),
    /// Sigmoid activation function.
    Sigmoid(Box<Expr>),
    /// Negation.
    Neg(Box<Expr>),
    /// Exponential function (e^x).
    Exp(Box<Expr>),
    /// Square root function.
    Sqrt(Box<Expr>),
    /// Step function (returns 1.0 if x >= 0.0, else 0.0), used for gradients.
    Step(Box<Expr>),
}

impl Expr {
    /// Render this AST node to WGSL shader source code.
    pub fn to_wgsl(&self, input_indexer: &dyn Fn(usize) -> String) -> String {
        match self {
            Expr::Input(idx) => input_indexer(*idx),
            Expr::Scalar(val) => {
                let s = format!("{:?}", val);
                if s.contains('.') || s.contains('e') {
                    s
                } else {
                    format!("{}.0", s)
                }
            }
            Expr::Add(lhs, rhs) => format!(
                "({} + {})",
                lhs.to_wgsl(input_indexer),
                rhs.to_wgsl(input_indexer)
            ),
            Expr::Mul(lhs, rhs) => format!(
                "({} * {})",
                lhs.to_wgsl(input_indexer),
                rhs.to_wgsl(input_indexer)
            ),
            Expr::Sub(lhs, rhs) => format!(
                "({} - {})",
                lhs.to_wgsl(input_indexer),
                rhs.to_wgsl(input_indexer)
            ),
            Expr::Div(lhs, rhs) => format!(
                "({} / {})",
                lhs.to_wgsl(input_indexer),
                rhs.to_wgsl(input_indexer)
            ),
            Expr::Relu(val) => format!("max(0.0, {})", val.to_wgsl(input_indexer)),
            Expr::Tanh(val) => format!("tanh({})", val.to_wgsl(input_indexer)),
            Expr::Sigmoid(val) => format!("(1.0 / (1.0 + exp(-({}))))", val.to_wgsl(input_indexer)),
            Expr::Neg(val) => format!("(-({}))", val.to_wgsl(input_indexer)),
            Expr::Exp(val) => format!("exp({})", val.to_wgsl(input_indexer)),
            Expr::Sqrt(val) => format!("sqrt({})", val.to_wgsl(input_indexer)),
            Expr::Step(val) => format!("step(0.0, {})", val.to_wgsl(input_indexer)),
        }
    }

    /// Serialize the AST node into a string representation for caching keys.
    pub fn serialize(&self) -> String {
        match self {
            Expr::Input(idx) => format!("Input({})", idx),
            Expr::Scalar(val) => format!("Scalar({})", val),
            Expr::Add(lhs, rhs) => format!("Add({},{})", lhs.serialize(), rhs.serialize()),
            Expr::Mul(lhs, rhs) => format!("Mul({},{})", lhs.serialize(), rhs.serialize()),
            Expr::Sub(lhs, rhs) => format!("Sub({},{})", lhs.serialize(), rhs.serialize()),
            Expr::Div(lhs, rhs) => format!("Div({},{})", lhs.serialize(), rhs.serialize()),
            Expr::Relu(val) => format!("Relu({})", val.serialize()),
            Expr::Tanh(val) => format!("Tanh({})", val.serialize()),
            Expr::Sigmoid(val) => format!("Sigmoid({})", val.serialize()),
            Expr::Neg(val) => format!("Neg({})", val.serialize()),
            Expr::Exp(val) => format!("Exp({})", val.serialize()),
            Expr::Sqrt(val) => format!("Sqrt({})", val.serialize()),
            Expr::Step(val) => format!("Step({})", val.serialize()),
        }
    }
}

/// Evaluate this AST on the CPU for the given input buffer views at a specific output index.
pub fn evaluate_ast(
    expr: &Expr,
    inputs: &[&[f32]],
    input_shapes: &[Shape],
    output_shape: &Shape,
    output_idx: usize,
) -> f32 {
    match expr {
        Expr::Input(idx) => {
            let input_slice = inputs[*idx];
            let in_shape = &input_shapes[*idx];
            if in_shape == output_shape {
                input_slice[output_idx]
            } else {
                let out_dims = output_shape.dims();
                let mut out_coord = vec![0; out_dims.len()];
                let mut temp = output_idx;
                for i in (0..out_dims.len()).rev() {
                    out_coord[i] = temp % out_dims[i];
                    temp /= out_dims[i];
                }

                let in_dims = in_shape.dims();
                let mut in_coord = vec![0; in_dims.len()];
                for i in 0..in_dims.len() {
                    let out_axis = out_dims.len() - in_dims.len() + i;
                    if in_dims[i] != 1 {
                        in_coord[i] = out_coord[out_axis];
                    }
                }

                let mut in_idx = 0;
                let mut stride = 1;
                for i in (0..in_dims.len()).rev() {
                    in_idx += in_coord[i] * stride;
                    stride *= in_dims[i];
                }
                input_slice[in_idx]
            }
        }
        Expr::Scalar(val) => *val,
        Expr::Add(lhs, rhs) => {
            evaluate_ast(lhs, inputs, input_shapes, output_shape, output_idx)
                + evaluate_ast(rhs, inputs, input_shapes, output_shape, output_idx)
        }
        Expr::Mul(lhs, rhs) => {
            evaluate_ast(lhs, inputs, input_shapes, output_shape, output_idx)
                * evaluate_ast(rhs, inputs, input_shapes, output_shape, output_idx)
        }
        Expr::Sub(lhs, rhs) => {
            evaluate_ast(lhs, inputs, input_shapes, output_shape, output_idx)
                - evaluate_ast(rhs, inputs, input_shapes, output_shape, output_idx)
        }
        Expr::Div(lhs, rhs) => {
            evaluate_ast(lhs, inputs, input_shapes, output_shape, output_idx)
                / evaluate_ast(rhs, inputs, input_shapes, output_shape, output_idx)
        }
        Expr::Relu(val) => {
            let v = evaluate_ast(val, inputs, input_shapes, output_shape, output_idx);
            if v > 0.0 {
                v
            } else {
                0.0
            }
        }
        Expr::Tanh(val) => evaluate_ast(val, inputs, input_shapes, output_shape, output_idx).tanh(),
        Expr::Sigmoid(val) => {
            let v = evaluate_ast(val, inputs, input_shapes, output_shape, output_idx);
            1.0 / (1.0 + (-v).exp())
        }
        Expr::Neg(val) => -evaluate_ast(val, inputs, input_shapes, output_shape, output_idx),
        Expr::Exp(val) => evaluate_ast(val, inputs, input_shapes, output_shape, output_idx).exp(),
        Expr::Sqrt(val) => evaluate_ast(val, inputs, input_shapes, output_shape, output_idx).sqrt(),
        Expr::Step(val) => {
            let v = evaluate_ast(val, inputs, input_shapes, output_shape, output_idx);
            if v >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Compute a unique hash of this AST based on structure, inputs, and output types.
pub fn hash_ast(expr: &Expr, input_shapes: &[Shape], dtype: Dtype) -> String {
    let mut hasher = Sha256::new();
    hasher.update(expr.serialize().as_bytes());
    for shape in input_shapes {
        hasher.update(format!("{:?}", shape.dims()).as_bytes());
    }
    hasher.update(format!("{:?}", dtype).as_bytes());
    let result = hasher.finalize();
    result
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_wgsl() {
        let expr = Expr::Relu(Box::new(Expr::Add(
            Box::new(Expr::Input(0)),
            Box::new(Expr::Scalar(2.5)),
        )));
        let indexer = |idx| format!("in{}", idx);
        assert_eq!(expr.to_wgsl(&indexer), "max(0.0, (in0 + 2.5))");
    }

    #[test]
    fn test_step_wgsl() {
        let expr = Expr::Step(Box::new(Expr::Input(0)));
        let indexer = |idx| format!("in{}", idx);
        assert_eq!(expr.to_wgsl(&indexer), "step(0.0, in0)");
    }
}
