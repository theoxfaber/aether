use aether::{Device, Graph, Shape};

fn main() {
    println!("=== Aether V2: Dynamic AST Codegen & Broadcasting Example ===");
    let graph = Graph::new();

    let a = graph.tensor(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    let b = graph.tensor(vec![0.5, 1.5], Shape::new(vec![1, 2]));

    // Perform fused broadcast: (a + b) * a
    let c = a.add(b).mul(a);

    println!("Running computation on Wgpu GPU...");
    let result = c.run(Device::Wgpu).unwrap();

    println!("Input A:\n  [1.0, 2.0]\n  [3.0, 4.0]");
    println!("Input B (to be broadcast):\n  [0.5, 1.5]");
    println!("Result (A + B) * A:\n  {:?}", result.data());

    let expected = vec![
        (1.0 + 0.5) * 1.0,
        (2.0 + 1.5) * 2.0,
        (3.0 + 0.5) * 3.0,
        (4.0 + 1.5) * 4.0,
    ];
    println!("Expected result:\n  {:?}", expected);

    assert_eq!(result.data(), &expected[..]);
    println!("Success! Verification passed.");
}
