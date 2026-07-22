//! Build script for co-bsim4 (R2).
//!
//! Two jobs, both driven off the *unmodified* vendored Berkeley BSIM4.5 tree at
//! `circuitopt/compact_models/bsim4/native_src/` (architecture decision D-a):
//!
//! 1. Compile the same C translation units the Python `native.py` backend builds
//!    (`native.py` `_SOURCES`) — minus `host.c`, which is ported to Rust — into a
//!    static archive, with flags aligned to `native.py`
//!    (`-O2 -std=c99 -fPIC -Wno-error=implicit-function-declaration`).
//! 2. Run bindgen over the same header include set `host.c` uses, so the Rust
//!    host in `src/lib.rs` shares an identical struct layout / ABI with the C.

use std::env;
use std::path::{Path, PathBuf};

/// Vendor C translation units compiled here. This is exactly `native.py`'s
/// `_SOURCES` with `host.c` removed (host is the Rust port).
const VENDOR_MODEL_SOURCES: &[&str] = &[
    "b4v5.c",
    "b4v5par.c",
    "b4v5mpar.c",
    "b4v5set.c",
    "b4v5temp.c",
    "b4v5ld.c",
    "b4v5acld.c",
    "b4v5noi.c",
    "b4v5geo.c",
];

/// `std::fs::canonicalize` returns `\\?\`-prefixed extended-length paths on
/// Windows; cl.exe's C1 front end rejects them for *source* files (C1083:
/// cannot open '\\b4v5.c'). Strip the prefix; a no-op everywhere else.
fn strip_extended_length_prefix(path: PathBuf) -> PathBuf {
    match path.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(rest) => PathBuf::from(rest),
        None => path,
    }
}

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_src = strip_extended_length_prefix(
        crate_dir
            .join("../../../circuitopt/compact_models/bsim4/native_src")
            .canonicalize()
            .expect("locate circuitopt/compact_models/bsim4/native_src (vendor tree)"),
    );
    let vendor = native_src.join("vendor");
    let include_dir = vendor.join("include");
    let model_dir = vendor.join("bsim4v5");
    let support_dir = vendor.join("support");

    // Are we compiling for the MSVC target ABI (x86_64-pc-windows-msvc)? Cargo
    // sets CARGO_CFG_TARGET_ENV per *target* for build scripts, so this branches
    // correctly under cross-compilation and is empty ("") on macOS, "gnu"/"musl"
    // on Linux, and "gnu" on the windows-gnu (MinGW) ABI — i.e. only the true
    // MSVC target takes the Windows path below; everything else keeps the exact
    // historical clang/gcc invocation.
    let is_msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    // Crate-local shim directory, OUTSIDE the frozen vendor tree. On MSVC only,
    // it is prepended to the include path so its `ngspice/config.h` shadows the
    // vendored, POSIX-configured one (the vendored headers already carry
    // `_MSC_VER` branches; the only thing pulling in <unistd.h>/<strings.h>/
    // <dirent.h> is that config.h advertising HAVE_UNISTD_H etc.). The vendor
    // tree is never modified, and this directory is never on the include path
    // for non-MSVC targets, so macOS/Linux builds are bit-for-bit unchanged.
    let msvc_shim = crate_dir.join("msvc_shim");

    // ---- 1. compile the vendored Berkeley BSIM4.5 C -----------------------
    let mut build = cc::Build::new();
    // MSVC: the shim include must precede the vendor include so its config.h
    // wins the `#include "ngspice/config.h"` lookup. Added only for MSVC, so the
    // unix include order is untouched.
    if is_msvc {
        build.include(&msvc_shim);
    }
    build
        .include(&include_dir)
        .include(&model_dir)
        .opt_level(2)
        .pic(true)
        .warnings(false);
    if is_msvc {
        // cl.exe has no `-std=c99` switch, and — crucially — its *default* C mode
        // keeps implicit-function-declaration a warning (C4013), which is exactly
        // the tolerance the unix `-Wno-error=implicit-function-declaration` flag
        // buys; a `/std:c11`/`/std:c17` switch would instead push it toward an
        // error. So MSVC deliberately gets neither unix flag and uses its
        // permissive default. `.pic()`/`.opt_level()` are translated by `cc`.
    } else {
        build
            .flag("-std=c99")
            // clang 16+ promotes implicit-function-declaration to an error in C99
            // mode; the unmodified Berkeley sources rely on implicit libc decls.
            // Keep it a warning, exactly like native.py.
            .flag("-Wno-error=implicit-function-declaration");
    }
    for name in VENDOR_MODEL_SOURCES {
        build.file(model_dir.join(name));
    }
    build.file(support_dir.join("devsup.c"));
    build.compile("co_bsim4_vendor");

    // ---- 2. bindgen the shared header set ---------------------------------
    // bindgen always parses with libclang (never cl.exe), so `-std=c99` is a
    // valid clang arg on every host and stays. On MSVC the same config.h shim is
    // prepended so libclang, too, skips the POSIX-only includes.
    let wrapper = crate_dir.join("csrc/wrapper.h");
    let mut builder = bindgen::Builder::default().header(wrapper.to_string_lossy());
    if is_msvc {
        builder = builder.clang_arg(format!("-I{}", msvc_shim.display()));
    }
    let bindings = builder
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", model_dir.display()))
        .clang_arg("-std=c99")
        // Only surface the types/functions/vars the Rust host actually touches
        // (plus their transitive dependencies); the ngspice headers pull in the
        // whole simulator otherwise.
        .allowlist_function("BSIM4v5(setup|temp|load|acLoad|noise|param|mParam)")
        .allowlist_type("BSIM4v5model")
        .allowlist_type("BSIM4v5instance")
        .allowlist_type("CKTcircuit")
        .allowlist_type("CKTnode")
        .allowlist_type("IFparm")
        .allowlist_type("IFvalue")
        .allowlist_type("IFfrontEnd")
        .allowlist_type("Ndata")
        .allowlist_type("NOISEAN")
        .allowlist_type("circ")
        .allowlist_type("TSKtask")
        .allowlist_type("GENmodel")
        .allowlist_type("GENinstance")
        .allowlist_type("SMPmatrix")
        .allowlist_type("bsim4v5SizeDependParam")
        .allowlist_var("BSIM4v5pTable")
        .allowlist_var("BSIM4v5mPTable")
        .allowlist_var("BSIM4v5pTSize")
        .allowlist_var("BSIM4v5mPTSize")
        // Keep the generated file lean and robust: no per-type layout unit
        // tests (layout correctness comes from libclang parsing the same
        // headers the C is compiled from).
        .layout_tests(false)
        .derive_default(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen the BSIM4.5 vendor headers");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindgen output");

    // ---- rerun triggers ---------------------------------------------------
    // The vendor tree is frozen, but a change to any source or header must
    // rebuild both the archive and the bindings.
    println!("cargo:rerun-if-changed={}", wrapper.display());
    rerun_if_changed_tree(&model_dir);
    rerun_if_changed_tree(&include_dir.join("ngspice"));
    println!(
        "cargo:rerun-if-changed={}",
        support_dir.join("devsup.c").display()
    );
}

/// Emit `cargo:rerun-if-changed` for every file under `dir` (non-recursive glob
/// of `.c`/`.h`), plus the directory itself.
fn rerun_if_changed_tree(dir: &Path) {
    println!("cargo:rerun-if-changed={}", dir.display());
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            match path.extension().and_then(|e| e.to_str()) {
                Some("c") | Some("h") => {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
                _ => {}
            }
        }
    }
}
