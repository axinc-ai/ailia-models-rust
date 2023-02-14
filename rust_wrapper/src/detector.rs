use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

use ailia_sys::*;

use image::RgbaImage;

use opencv::core::Mat;
use opencv::prelude::MatTraitConstManual;

use crate::network::Network;
use crate::AiliaError;

pub use ailia_sys::AILIA_NETWORK_IMAGE_FORMAT_BGR;
pub use ailia_sys::AILIA_NETWORK_IMAGE_FORMAT_GRAY;
pub use ailia_sys::AILIA_NETWORK_IMAGE_FORMAT_GRAY_EQUALIZE;
pub use ailia_sys::AILIA_NETWORK_IMAGE_FORMAT_RGB;

pub use ailia_sys::AILIA_NETWORK_IMAGE_CHANNEL_FIRST;
pub use ailia_sys::AILIA_NETWORK_IMAGE_CHANNEL_LAST;

pub use ailia_sys::AILIA_NETWORK_IMAGE_RANGE_IMAGENET;
pub use ailia_sys::AILIA_NETWORK_IMAGE_RANGE_SIGNED_FP32;
pub use ailia_sys::AILIA_NETWORK_IMAGE_RANGE_SIGNED_INT8;
pub use ailia_sys::AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_FP32;
pub use ailia_sys::AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_INT8;

pub use ailia_sys::AILIA_DETECTOR_OBJECT_VERSION;

pub use ailia_sys::AILIA_DETECTOR_ALGORITHM_SSD;
pub use ailia_sys::AILIA_DETECTOR_ALGORITHM_YOLOV1;
pub use ailia_sys::AILIA_DETECTOR_ALGORITHM_YOLOV2;
pub use ailia_sys::AILIA_DETECTOR_ALGORITHM_YOLOV3;
pub use ailia_sys::AILIA_DETECTOR_ALGORITHM_YOLOV4;
pub use ailia_sys::AILIA_DETECTOR_ALGORITHM_YOLOX;

pub use ailia_sys::AILIA_DETECTOR_FLAG_NORMAL;

// TODO:Option以外は設定されていない場合buildを呼べないようにする(型によって制限をかける)
#[derive(Clone, Copy, Debug, Default)]
pub struct DetectorBuilder<P>
where
    P: AsRef<Path> + Default + Debug,
{
    prototxt: P,
    onnx: P,
    env_id: Option<i32>,
    num_threads: Option<i32>,
    format: Option<u32>,
    channel: Option<u32>,
    range: Option<u32>,
    algorithm: u32,
    category_count: u32,
    flags: Option<u32>,
}

impl<P: AsRef<Path> + Default + Debug> DetectorBuilder<P> {
    crate::impl_non_option!(prototxt, P);
    crate::impl_non_option!(onnx, P);
    crate::impl_option!(env_id, i32);
    crate::impl_option!(num_threads, i32);
    crate::impl_option!(format, u32);
    crate::impl_option!(channel, u32);
    crate::impl_option!(range, u32);
    crate::impl_non_option!(algorithm, u32);
    crate::impl_non_option!(category_count, u32);
    crate::impl_option!(flags, u32);

    pub fn build(self) -> Result<Detector, AiliaError> {
        let net = Network::ailia_create(
            self.env_id.unwrap_or(AILIA_ENVIRONMENT_ID_AUTO),
            self.num_threads
                .unwrap_or_else(|| AILIA_MULTITHREAD_AUTO.try_into().unwrap()),
        )?;
        net.open_stream_file_a(self.prototxt)?;
        net.open_weight_file_a(self.onnx)?;
        Detector::new(
            net,
            self.format.unwrap_or(AILIA_NETWORK_IMAGE_FORMAT_RGB),
            self.channel.unwrap_or(AILIA_NETWORK_IMAGE_CHANNEL_FIRST),
            self.range
                .unwrap_or(AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_INT8),
            self.algorithm,
            self.category_count,
            self.flags.unwrap_or(AILIA_DETECTOR_FLAG_NORMAL),
        )
    }
}

// TODO: Builder構造体を作ってNetworkを直接newしなくてもDetectorを作れるようにする
pub struct Detector {
    inner: NonNull<AILIADetector>,
    net: Network,
}

impl Detector {
    pub fn new(
        net: Network,
        format: u32,
        channel: u32,
        range: u32,
        algorithm: u32,
        category_count: u32,
        flags: u32,
    ) -> Result<Self, AiliaError> {
        let mut ptr: *mut AILIADetector = std::ptr::null::<AILIADetector>() as *mut _;
        unsafe {
            match ailiaCreateDetector(
                (&mut ptr) as *mut *mut _,
                net.as_ptr() as *mut _,
                format,
                channel,
                range,
                algorithm,
                category_count,
                flags,
            ) {
                0 => Ok(Self {
                    inner: NonNull::new_unchecked(ptr),
                    net,
                }),
                i => Err(i.into()),
            }
        }
    }

    pub fn set_input_shape(&self, width: u32, height: u32) -> Result<(), AiliaError> {
        crate::invoke_ailia_fn_result!(ailiaDetectorSetInputShape, self.as_ptr(), width, height);
    }

    pub fn predict<T>(
        &self,
        image_ptr: *const T,
        stride: u32,
        width: u32,
        height: u32,
        format: u32,
        threshold: f32,
        iou: f32,
    ) -> Result<Vec<Object>, AiliaError> {
        self.compute(image_ptr, stride, width, height, format, threshold, iou)?;
        let num = self.get_object_count()?;
        let mut objes = Vec::with_capacity(num.try_into().unwrap());
        for idx in 0..num {
            objes.push(self.get_object(idx, AILIA_DETECTOR_OBJECT_VERSION)?);
        }
        Ok(objes)
    }

    // DynamicImage or ImageBufferを入力にするようにする
    pub fn predict_image(
        &self,
        image: RgbaImage,
        threshold: f32,
        iou: f32,
    ) -> Result<Vec<Object>, AiliaError> {
        self.predict(
            image.as_ptr() as *const _,
            image.width() * 4,
            image.width(),
            image.height(),
            AILIA_IMAGE_FORMAT_RGBA,
            threshold,
            iou,
        )
    }

    /// RGBAを入力してください
    pub fn predict_opencv_mat(
        &self,
        image: &Mat,
        threshold: f32,
        iou: f32,
    ) -> Result<Vec<Object>, AiliaError> {
        let size = image.size().expect("cannot get image size");
        self.predict(
            image.data(),
            (size.width * 4)
                .try_into()
                .expect("can't convert image.width to usize"),
            size.width
                .try_into()
                .expect("can't convert image.width to usize"),
            size.height
                .try_into()
                .expect("can't convert image.height to usize"),
            AILIA_IMAGE_FORMAT_RGBA,
            threshold,
            iou,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute<T>(
        &self,
        image_ptr: *const T,
        stride: u32,
        width: u32,
        height: u32,
        format: u32,
        threshold: f32,
        iou: f32,
    ) -> Result<(), AiliaError> {
        crate::invoke_ailia_fn_result!(
            ailiaDetectorCompute,
            self.as_ptr(),
            image_ptr as *const std::os::raw::c_void,
            stride,
            width,
            height,
            format,
            threshold,
            iou
        );
    }

    // DynamicImage or ImageBufferを入力にするようにする
    /// Image crateのRgbaImageを入力として計算する
    pub fn compute_image(
        &self,
        image: &RgbaImage,
        threshold: f32,
        iou: f32,
    ) -> Result<(), AiliaError> {
        self.compute(
            image.as_ptr() as *const std::os::raw::c_void,
            image.width() * 4,
            image.width(),
            image.height(),
            AILIA_IMAGE_FORMAT_RGBA,
            threshold,
            iou,
        )
    }

    /// from opencv Mat RGBAで入力してください
    pub fn compute_opencv_mat(
        &self,
        image: &Mat,
        threshold: f32,
        iou: f32,
    ) -> Result<(), AiliaError> {
        let size = image.size().unwrap();
        self.compute(
            image.data() as *const std::os::raw::c_void,
            (size.width * 4).try_into().unwrap(),
            size.width.try_into().unwrap(),
            size.height.try_into().unwrap(),
            AILIA_IMAGE_FORMAT_RGBA,
            threshold,
            iou,
        )
    }

    fn as_ptr(&self) -> *mut AILIADetector {
        self.inner.as_ptr()
    }

    pub fn get_object(&self, idx: u32, version: u32) -> Result<Object, AiliaError> {
        Object::get_object(self, idx, version)
    }

    pub fn get_object_count(&self) -> Result<u32, AiliaError> {
        let mut res = 0;
        match unsafe { ailiaDetectorGetObjectCount(self.as_ptr(), &mut res as *mut u32) } {
            0 => Ok(res),
            i => Err(i.into()),
        }
    }
}

impl Drop for Detector {
    fn drop(&mut self) {
        unsafe {
            ailiaDestroyDetector(self.inner.as_ptr());
        }
    }
}

impl Deref for Detector {
    type Target = Network;
    fn deref(&self) -> &Self::Target {
        &self.net
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Object {
    pub category: u32,
    pub prob: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Object {
    fn get_object(detector: &Detector, idx: u32, version: u32) -> Result<Self, AiliaError> {
        // let ptr: *mut AILIADetectorObject = std::ptr::null::<AILIADetectorObject>() as *mut _;
        let obj: MaybeUninit<AILIADetectorObject> = MaybeUninit::uninit();
        let ptr = obj.as_ptr() as *mut AILIADetectorObject;
        unsafe {
            match ailiaDetectorGetObject(detector.as_ptr(), ptr, idx, version) {
                0 => {
                    let obj = *ptr;
                    Ok(Self {
                        category: obj.category,
                        prob: obj.prob,
                        x: obj.x,
                        y: obj.y,
                        w: obj.w,
                        h: obj.h,
                    })
                }
                i => Err(i.into()),
            }
        }
    }
}
