extern crate bindgen;

use std::{env, path::PathBuf};

fn main() {
    let ailia_path = env::var("AILIA_BIN_DIR").expect("Please specify AILIA_BIN_DIR");
    let ailia_include_dir = env::var("AILIA_INC_DIR").expect("Please set AILIA_INC_DIR");

    println!("cargo:rustc-link-search=native={}", ailia_path);
    println!("cargo:rustc-link-lib=dylib=ailia");
    println!("cargo:rustc-link-lib=dylib=ailia_pose_estimate");
    println!("cargo:rerun-if-changed=wrapper.h");
    
    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", ailia_include_dir))
        .header("wrapper.h")
        .size_t_is_usize(true)
        .rustfmt_bindings(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to bind ailia");
    let out_path = PathBuf::from("src");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
