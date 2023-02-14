#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!("bindings.rs");

// #[cfg(test)]
// mod test {
//     use super::*;
//     // #[test]
//     // fn load_ailia() {
//     //     let network: MaybeUninit<AILIANetwork> = MaybeUninit::zeroed();
//     //     let status = unsafe {
//     //         ailiaCreate(network.as_ptr() as *mut *mut _, -1, 1)
//     //     };
//     //     assert_eq!(status, 0);
//     // }
//
//     // #[test]
//     // fn load_ailia_prototxt() {
//     //     let mut network: *const AILIANetwork = std::ptr::null();
//     //     let ptr_ptr_net = (&mut (network) as *mut _) as *mut *mut AILIANetwork;
//     //     let status = unsafe {
//     //         ailiaCreate(ptr_ptr_net, -1, 1)
//     //     };
//     //     assert_eq!(status, 0);
//     //     let path = "/home/bokutotu/HDD/Work/ailia/jenkins/data/layer/l/prototxt/affine.prototxt";
//     //     let path = path.as_ptr() as *const ::std::os::raw::c_char;
//     //     let status = unsafe {
//     //         ailiaOpenStreamFileA(network as *mut _, path)
//     //     };
//     //     assert_eq!(status, 0);
//     // }
//
//     #[test]
//     fn load_ailia_prototxt_weight() {
//         let mut network: *const AILIANetwork = std::ptr::null();
//         let ptr_ptr_net = (&mut (network) as *mut _) as *mut *mut AILIANetwork;
//         let status = unsafe {
//             ailiaCreate(ptr_ptr_net, -1, 1)
//         };
//         assert_eq!(status, 0);
//         let path = "/home/bokutotu/HDD/Work/ailia/jenkins/data/models/mobilenetv2_1.0.opt.onnx.prototxt";
//         let path = path.as_ptr() as *const ::std::os::raw::c_char;
//         let status = unsafe {
//             ailiaOpenStreamFileA(network as *mut _, path)
//         };
//         assert_eq!(status, 0);
//         // dbg!(status);
//         // let path = "/home/bokutotu/HDD/Work/ailia/jenkins/data/models/mobilenetv2_1.0.opt.onnx";
//         // let path = path.as_ptr() as *const ::std::os::raw::c_char;
//         // let status = unsafe {
//         //     ailiaOpenWeightFileA(network as *mut _, path)
//         // };
//         // dbg!(status);
//         // assert_eq!(status, 0);
//     }
// }
