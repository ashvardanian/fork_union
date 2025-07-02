fn main() -> Result<(), cc::Error> {
    let mut build = cc::Build::new();

    build
        .cpp(true) // Enable C++ support
        .file("c/lib.cpp")
        .include("include")
        .define("FU_ENABLE_NUMA", "0")
        .opt_level(3) // Set optimization level to 2
        .flag("-pedantic") // Ensure strict compliance with the C standard
        .flag("-std=c++17") // Specify C++ standard
        .warnings(false);

    if let Err(e) = build.try_compile("fork_union") {
        print!("cargo:warning={}", e);
        return Err(e);
    }

    println!("cargo:rerun-if-changed=c/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/fork_union.h");
    println!("cargo:rerun-if-changed=include/fork_union.hpp");
    Ok(())
}
