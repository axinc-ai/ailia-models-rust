#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!("bindings.rs");

#[cfg(test)]
mod test {
    use std::ffi::CString;

    use super::*;

    #[test]
    fn load_ailia_prototxt_weight() {
        let mut network: *const AILIANetwork = std::ptr::null();
        let ptr_ptr_net = (&mut (network) as *mut _) as *mut *mut AILIANetwork;
        let status = unsafe {
            ailiaCreate(ptr_ptr_net, -1, 1)
        };
        assert_eq!(status, 0);
        let path = "/home/bokutotu/HDD/Work/rust_wrapper/yolox_s.opt.onnx.prototxt".to_string();
        let path = CString::new(path).unwrap().as_ptr();
        let status = unsafe {
            ailiaOpenStreamFileA(network as *mut _, path)
        };
        assert_eq!(status, 0);
        let path = "/home/bokutotu/HDD/Work/rust_wrapper/yolox_s.opt.onnx".to_string();
        let path = CString::new(path).unwrap().as_ptr();
        let status = unsafe {
            ailiaOpenWeightFileA(network as *mut _, path)
        };
        assert_eq!(status, 0);
    }

    #[test]
    fn env() {
        let env: *const AILIAEnvironment = std::ptr::null();
        let mut env = env as *mut AILIAEnvironment;
        let res = unsafe { ailiaGetEnvironment((&mut env) as *mut *mut AILIAEnvironment, 0, 2) };
        assert_eq!(res, 0);
        let id = unsafe { (*env).id };
        dbg!(id);
    }
}
