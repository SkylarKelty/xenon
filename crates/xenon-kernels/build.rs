//! Build script for xenon-kernels.
//!
//! Compiles every `src/cu/*.cu` via `nvcc` into a single static library that
//! the rest of the Rust workspace links against, plus the CUDA runtime.
//!
//! Env overrides:
//!   NVCC       - path to nvcc (default: auto-detect /usr/local/cuda*)
//!   CUDA_LIB   - path to libcudart.so (default: /usr/local/cuda/lib64)
//!   XENON_ARCH - arch target (default: sm_120a for Blackwell)

use std::{env, path::PathBuf, process::Command};

fn find_nvcc() -> PathBuf {
    if let Ok(p) = env::var("NVCC") {
        return PathBuf::from(p);
    }
    for c in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda-12.9/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
    ] {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("nvcc")
}

fn main() {
    let nvcc = find_nvcc();
    let arch = env::var("XENON_ARCH").unwrap_or_else(|_| "sm_120a".to_string());
    let cuda_lib = env::var("CUDA_LIB").unwrap_or_else(|_| "/usr/local/cuda/lib64".to_string());

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = manifest.join("src/cu");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    // Re-run if a new .cu file appears (the per-file rerun-if-changed below
    // doesn't cover directory contents).
    println!("cargo:rerun-if-changed={}", src_dir.display());

    // Enumerate .cu sources.
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&src_dir)
        .expect("read src/cu")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("cu"))
        .collect();
    sources.sort();

    if sources.is_empty() {
        panic!("no .cu sources found in {}", src_dir.display());
    }

    let mut objs = Vec::new();
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
        let stem = src.file_stem().unwrap().to_string_lossy().into_owned();
        let obj = out_dir.join(format!("{stem}.o"));

        let status = Command::new(&nvcc)
            .args([
                "-c",
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-Xcompiler",
                "-fPIC,-Wall,-Wextra",
            ])
            .args(["-gencode", &format!("arch=compute_{a},code=sm_{a}",
                                        a = arch.trim_start_matches("sm_"))])
            .arg(src)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("invoke nvcc");
        if !status.success() {
            panic!("nvcc failed for {}", src.display());
        }
        objs.push(obj);
    }

    // Archive into a single static lib.
    let lib_path = out_dir.join("libxenon_kernels_impl.a");
    let _ = std::fs::remove_file(&lib_path);
    let status = Command::new("ar")
        .arg("crs")
        .arg(&lib_path)
        .args(&objs)
        .status()
        .expect("invoke ar");
    if !status.success() {
        panic!("ar failed");
    }

    // Link instructions.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_LIB");
    println!("cargo:rerun-if-env-changed=XENON_ARCH");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=xenon_kernels_impl");
    println!("cargo:rustc-link-search=native={cuda_lib}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // Bake rpath so binaries find libcudart/libcublasLt without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{cuda_lib}");
}
