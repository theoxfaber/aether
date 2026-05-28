use aether::{Device, Graph, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = Graph::new();

    let tensor_b = graph.tensor(vec![2.0, 0.0, 0.0, 2.0], Shape::new(vec![2, 2]));

    let result = graph
        .tensor(vec![1.0, -2.0, 3.0, 4.0], Shape::new(vec![2, 2]))
        .matmul(tensor_b)
        .relu()
        .run(Device::Cpu)?;

    println!("Matrix A: [[1.0, -2.0], [3.0, 4.0]]");
    println!("Matrix B: [[2.0,  0.0], [0.0,  2.0]]");
    println!("Computed output (A * B followed by ReLU):");
    println!("Data: {:?}", result.data());
    println!("Shape: {:?}", result.shape().dims());

    let expected = vec![2.0, 0.0, 6.0, 8.0];
    assert_eq!(result.data(), &expected[..]);
    println!("Validation successful! Numerical correctness verified.");

    Ok(())
}
