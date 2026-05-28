use aether::{Device, Graph, Shape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = Graph::new();

    let b = graph.tensor(vec![2.0f32, 0.0, 0.0, 2.0], Shape::new(vec![2, 2]));

    let result = graph
        .tensor(vec![1.0f32, -2.0, 3.0, 4.0], Shape::new(vec![2, 2]))
        .matmul(b)
        .relu()
        .run(Device::Wgpu)?;

    println!("GPU result: {:?}", result.data());
    println!("Shape: {:?}", result.shape().dims());

    let expected = vec![2.0f32, 0.0, 6.0, 8.0];
    assert_eq!(result.data(), &expected[..], "GPU result must match CPU");
    println!("GPU correctness verified!");
    Ok(())
}
