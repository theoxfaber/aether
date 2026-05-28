fn main() {
    // Link Apple Accelerate framework for BLAS-optimized matmul
    if std::env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
