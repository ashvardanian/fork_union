fn main() -> Result<(), cc::Error> {
    let mut build = cc::Build::new();

    build
        .cpp(true) // Enable C++ support
        .file("c/lib.cpp")
        .include("include")
        .define("FU_ENABLE_NUMA", "0")
        .opt_level(2) // Optimize compiled C++ to -O2
        .warnings(false);

    // Platform-specific C++ standard flags
    if cfg!(target_env = "msvc") {
        build.flag("/std:c++17"); // MSVC flag for C++17
    } else {
        build.flag("-pedantic"); // GCC/Clang strict compliance
        build.flag("-std=c++17"); // GCC/Clang C++17 flag
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
