//! Bake the building rustc version into the extension so `engine_info()` can
//! report the exact toolchain that produced the wheel.
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=CO_RUSTC_VERSION={version}");
    println!("cargo:rerun-if-changed=build.rs");
}
