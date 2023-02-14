use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;

use num_traits::Num;

use ailia_sys::*;

use crate::AiliaError;

pub struct Network {
    inner: NonNull<AILIANetwork>,
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
    pub dim: u32,
}

impl Shape {
    pub fn num_elms(&self) -> u32 {
        self.x * self.y * self.z * self.w
    }
}

impl From<MaybeUninit<_AILIAShape>> for Shape {
    fn from(value: MaybeUninit<_AILIAShape>) -> Self {
        let shape = unsafe { *value.as_ptr() };
        Shape::from(shape)
    }
}

impl From<_AILIAShape> for Shape {
    fn from(value: _AILIAShape) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: value.w,
            dim: value.dim,
        }
    }
}

impl From<Shape> for _AILIAShape {
    fn from(value: Shape) -> Self {
        _AILIAShape {
            x: value.x,
            y: value.y,
            z: value.z,
            w: value.w,
            dim: value.dim,
        }
    }
}

// pythonのpredictメソッドは入力のデータ型が異なる場合があるため、使用できない。
// detecのopset16の場合などは入力がf32とint64があり、これをrustでやる場合は他に工夫が必要
impl Network {
    pub fn ailia_create(env_id: i32, num_threads: i32) -> Result<Self, AiliaError> {
        let ptr: *const AILIANetwork = std::ptr::null();
        let mut ptr = ptr as *mut AILIANetwork;
        let ptr_ptr = (&mut ptr) as *mut *mut AILIANetwork;
        unsafe {
            match ailiaCreate(ptr_ptr, env_id, num_threads) {
                0 => Ok(Self {
                    inner: NonNull::new_unchecked(ptr),
                }),
                i => Err(i.into()),
            }
        }
    }

    pub fn open_stream_file_a<P: AsRef<Path>>(&self, prototxt_path: P) -> Result<(), AiliaError> {
        let path_string = prototxt_path.as_ref().to_str().unwrap().to_string();
        let path_cstring = CString::new(path_string).unwrap();
        crate::invoke_ailia_fn_result!(
            ailiaOpenStreamFileA,
            self.inner.as_ptr(),
            path_cstring.as_ptr()
        );
    }

    pub fn open_weight_file_a<P: AsRef<Path>>(&self, model_path: P) -> Result<(), AiliaError> {
        let path_string = model_path.as_ref().to_str().unwrap().to_string();
        let path_cstring = CString::new(path_string).unwrap();
        crate::invoke_ailia_fn_result!(
            ailiaOpenWeightFileA,
            self.inner.as_ptr(),
            path_cstring.as_ptr()
        );
    }

    pub fn new<P: AsRef<Path>>(
        env_id: i32,
        num_threads: i32,
        prototxt_path: P,
        model_path: P,
    ) -> Result<Self, AiliaError> {
        let model = Network::ailia_create(env_id, num_threads)?;
        model.open_stream_file_a(prototxt_path)?;
        model.open_weight_file_a(model_path)?;
        Ok(model)
    }

    pub fn as_ptr(&self) -> *mut AILIANetwork {
        self.inner.as_ptr()
    }

    pub fn get_input_shape(&self) -> Result<Shape, AiliaError> {
        let ptr: MaybeUninit<AILIAShape> = MaybeUninit::uninit();
        match unsafe {
            ailiaGetInputShape(self.as_ptr(), ptr.as_ptr() as *mut _, AILIA_SHAPE_VERSION)
        } {
            0 => Ok(ptr.into()),
            i => Err(i.into()),
        }
    }

    pub fn get_blob_shape(&self, idx: u32) -> Result<Shape, AiliaError> {
        let mut ptr: MaybeUninit<AILIAShape> = MaybeUninit::uninit();
        match unsafe {
            ailiaGetBlobShape(self.as_ptr(), ptr.as_mut_ptr(), idx, AILIA_SHAPE_VERSION)
        } {
            0 => Ok(ptr.into()),
            i => Err(i.into()),
        }
    }

    pub fn get_output_shape(&self) -> Result<Shape, AiliaError> {
        let ptr: MaybeUninit<AILIAShape> = MaybeUninit::uninit();
        match unsafe {
            ailiaGetOutputShape(self.as_ptr(), ptr.as_ptr() as *mut _, AILIA_SHAPE_VERSION)
        } {
            0 => Ok(ptr.into()),
            i => Err(i.into()),
        }
    }

    pub fn find_blob_index_by_nane(&self, name: &str) -> Result<u32, AiliaError> {
        let mut idx: u32 = 0;
        let cstring = CString::new(name).unwrap();
        match unsafe { ailiaFindBlobIndexByName(self.as_ptr(), &mut idx, cstring.as_ptr()) } {
            0 => Ok(idx),
            i => Err(i.into()),
        }
    }

    pub fn get_input_blob_count(&self) -> Result<u32, AiliaError> {
        let mut count = 0;
        match unsafe { ailiaGetInputBlobCount(self.as_ptr(), &mut count as *mut _) } {
            0 => Ok(count),
            i => Err(i.into()),
        }
    }

    pub fn get_output_blob_count(&self) -> Result<u32, AiliaError> {
        let mut count = 0;
        match unsafe { ailiaGetOutputBlobCount(self.as_ptr(), &mut count as *mut _) } {
            0 => Ok(count),
            i => Err(i.into()),
        }
    }

    pub fn get_input_blob_index_by_index(&self, idx: u32) -> Result<u32, AiliaError> {
        let mut res = 0;
        match unsafe { ailiaGetBlobIndexByInputIndex(self.as_ptr(), &mut res as *mut _, idx) } {
            0 => Ok(res),
            i => Err(i.into()),
        }
    }

    pub fn get_output_blob_index_by_index(&self, idx: u32) -> Result<u32, AiliaError> {
        let mut res = 0;
        match unsafe { ailiaGetBlobIndexByOutputIndex(self.as_ptr(), &mut res as *mut _, idx) } {
            0 => Ok(res),
            i => Err(i.into()),
        }
    }

    pub fn find_blob_idx_by_name(&self, name: &str) -> Result<u32, AiliaError> {
        let mut idx = 0;
        let name = CString::new(name).unwrap();
        match unsafe {
            ailiaFindBlobIndexByName(self.as_ptr(), &mut idx as *mut u32, name.as_ptr())
        } {
            0 => Ok(idx),
            i => Err(i.into()),
        }
    }

    pub fn get_input_indexs(&self) -> Result<Vec<u32>, AiliaError> {
        let count = self.get_input_blob_count()?;
        let mut indexes = Vec::with_capacity(count.try_into().unwrap());
        for idx in 0..count {
            indexes.push(self.get_input_blob_index_by_index(idx)?);
        }
        Ok(indexes)
    }

    pub fn get_output_indexs(&self) -> Result<Vec<u32>, AiliaError> {
        let count = self.get_output_blob_count()?;
        let mut indexes = Vec::with_capacity(count.try_into().unwrap());
        for idx in 0..count {
            indexes.push(self.get_output_blob_index_by_index(idx)?);
        }
        Ok(indexes)
    }

    pub fn set_input_blob_shape_nd(&self, shape_v: Vec<u32>, idx: u32) -> Result<(), AiliaError> {
        crate::invoke_ailia_fn_result!(ailiaSetInputBlobShapeND, self.as_ptr(), shape_v.as_ptr(), shape_v.len() as u32, idx);
    }

    pub fn set_input_blob_shape(&self, shape: Shape, idx: u32) -> Result<(), AiliaError> {
        let shape: _AILIAShape = shape.into();
        crate::invoke_ailia_fn_result!(
            ailiaSetInputBlobShape,
            self.as_ptr(),
            &shape as *const _AILIAShape,
            idx,
            AILIA_SHAPE_VERSION
        );
    }

    pub fn set_input_data_blob<T>(
        &self,
        src: *const T,
        size: u32,
        idx: u32,
    ) -> Result<(), AiliaError> {
        crate::invoke_ailia_fn_result!(
            ailiaSetInputBlobData,
            self.as_ptr(),
            src as *const std::os::raw::c_void,
            size * std::mem::size_of::<T>() as u32,
            idx
        );
    }

    pub fn get_output_blob_by_index<T: Num>(&self, idx: u32) -> Result<Vec<T>, AiliaError> {
        let shape = self.get_blob_shape(idx)?;
        let num_elms = shape.num_elms();
        let mut res = Vec::with_capacity(num_elms.try_into().unwrap());
        for _ in 0..num_elms {
            res.push(T::zero())
        }
        match unsafe {
            ailiaGetBlobData(
                self.as_ptr(),
                res.as_mut_ptr() as *mut _,
                num_elms * std::mem::size_of::<T>() as u32,
                idx,
            )
        } {
            0 => Ok(res),
            i => Err(i.into()),
        }
    }

    pub fn update(&self) -> Result<(), AiliaError> {
        crate::invoke_ailia_fn_result!(ailiaUpdate, self.as_ptr());
    }

    // pub fn predict<t>(&self, input: vec<&[t]>) -> result<vec<vec<t>>, ailiaerror> {
    //     let input_indexes = self.get_input_indexs()?;
    //     if input_indexes.len() != input.len().try_into().unwrap() {
    //         panic!("network input length is different from input length");
    //     }
    //     for (idx, input_blob_idx) in input_indexes.into_iter().enumerate() {
    //         self.set_input_data_blob(input[idx as usize].as_ptr(), input[idx as usize].len().try_into().unwrap(), input_blob_idx)?;
    //     }
    //     let output_indexes = self.get_output_indexs()?;
    //     let mut res = vec::with_capacity(output_indexes.len());
    //     for idx in output_indexes {
    //         res.push(self.get_output_blob_by_index(idx)?);
    //     }
    //     ok(res)
    // }
    //
    // pub fn predict_single_input<t>(self, input: &[t]) -> result<vec<vec<t>>, ailiaerror> {
    //     self.predict(vec![input])
    // }

    pub fn ailia_predict<D, S>(
        &self,
        dest: *mut D,
        dest_size: u32,
        src: *const S,
        src_size: u32,
    ) -> Result<(), AiliaError> {
        crate::invoke_ailia_fn_result!(
            ailiaPredict,
            self.as_ptr(),
            dest as *mut std::os::raw::c_void,
            dest_size,
            src as *const std::os::raw::c_void,
            src_size
        );
    }

    pub fn set_input_shape(&self, shape: Shape) -> Result<(), AiliaError> {
        let shape: _AILIAShape = Into::into(shape);
        crate::invoke_ailia_fn_result!(
            ailiaSetInputShape,
            self.as_ptr(),
            &shape as *const _,
            AILIA_SHAPE_VERSION
        );
    }

    pub fn get_error_ditail<'a>(&'a self) -> &'a str {
        let char_ptr = unsafe { ailiaGetErrorDetail(self.as_ptr()) };
        unsafe { CStr::from_ptr(char_ptr).to_str().unwrap() }
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe { ailiaDestroy(self.inner.as_ptr() as *mut _) };
    }
}

#[test]
fn test() {
    let net = Network::ailia_create(-1, 1).unwrap();
    net.open_stream_file_a("./yolox_s.opt.onnx.prototxt")
        .unwrap();
    net.open_weight_file_a("./yolox_s.opt.onnx").unwrap()
}
