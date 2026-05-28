fn main() {
    // Link Apple Accelerate framework for BLAS-optimized matmul (macOS only)
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if std::env::var("CARGO_FEATURE_ACCELERATE").is_ok() && target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
