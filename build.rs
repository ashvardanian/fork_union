fn main() -> Result<(), cc::Error> {
    let mut build = cc::Build::new();

    // Detect features and target for NUMA enablement
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let numa_feature_enabled = std::env::var("CARGO_FEATURE_NUMA").is_ok();
    let enable_numa = target_os == "linux" && numa_feature_enabled;

    build
        .cpp(true) // Enable C++ support
        .file("c/lib.cpp")
        .include("include")
        .define("FU_ENABLE_NUMA", if enable_numa { "1" } else { "0" })
        .opt_level(2) // Optimize compiled C++ to -O2
        .warnings(false);

    // Platform-specific C++ standard flags
    if cfg!(target_env = "msvc") {
        build.flag("/std:c++17"); // MSVC flag for C++17
    } else {
        build.flag("-pedantic"); // GCC/Clang strict compliance
        build.flag("-std=c++17"); // GCC/Clang C++17 flag
    }

    if enable_numa {
        // Link against system libraries when NUMA is enabled on Linux
        println!("cargo:rustc-link-lib=numa");
        println!("cargo:rustc-link-lib=pthread");
    }

    if let Err(e) = build.try_compile("fork_union") {
        print!("cargo:warning={e}");
        return Err(e);
    }

    println!("cargo:rerun-if-changed=c/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/fork_union.h");
    println!("cargo:rerun-if-changed=include/fork_union.hpp");
    Ok(())
}
