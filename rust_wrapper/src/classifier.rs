use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

use crate::network::Network;
use crate::AiliaError;

use ailia_sys::*;

// TODO:Option以外は設定されていない場合buildを呼べないようにする(型によって制限をかける)
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassifierBuilder<P>
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
}

impl<P: AsRef<Path> + Default + Debug> ClassifierBuilder<P> {
    crate::impl_non_option!(prototxt, P);
    crate::impl_non_option!(onnx, P);
    crate::impl_option!(env_id, i32);
    crate::impl_option!(num_threads, i32);
    crate::impl_option!(format, u32);
    crate::impl_option!(channel, u32);
    crate::impl_option!(range, u32);

    pub fn build(self) -> Result<Classifier, AiliaError> {
        let net = Network::ailia_create(
            self.env_id.unwrap_or(AILIA_ENVIRONMENT_ID_AUTO),
            self.num_threads
                .unwrap_or_else(|| AILIA_MULTITHREAD_AUTO.try_into().unwrap()),
        )?;
        net.open_stream_file_a(self.prototxt)?;
        net.open_weight_file_a(self.onnx)?;
        Classifier::new(
            net,
            self.format.unwrap_or(AILIA_NETWORK_IMAGE_FORMAT_RGB),
            self.channel.unwrap_or(AILIA_NETWORK_IMAGE_CHANNEL_FIRST),
            self.range
                .unwrap_or(AILIA_NETWORK_IMAGE_RANGE_UNSIGNED_INT8),
        )
    }
}

// TODO: Builder構造体を作ってNetworkを直接newしなくてもClassifierを作れるようにする
pub struct Classifier {
    inner: NonNull<AILIAClassifier>,
    net: Network,
}

#[derive(Clone, Debug, Copy)]
pub struct Class {
    pub category: i32,
    pub prob: f32,
}

impl Deref for Classifier {
    type Target = Network;
    fn deref(&self) -> &Self::Target {
        &self.net
    }
}

impl Drop for Classifier {
    fn drop(&mut self) {
        unsafe { ailiaDestroyClassifier(self.inner.as_ptr()) };
    }
}

impl From<MaybeUninit<AILIAClassifierClass>> for Class {
    fn from(value: MaybeUninit<AILIAClassifierClass>) -> Self {
        let ptr = value.as_ptr() as *const _AILIAClassifierClass;
        unsafe {
            Self {
                category: (*ptr).category,
                prob: (*ptr).prob,
            }
        }
    }
}

impl Classifier {
    pub fn new(net: Network, format: u32, channel: u32, range: u32) -> Result<Self, AiliaError> {
        let mut ptr: *mut AILIAClassifier = std::ptr::null::<AILIAClassifier>() as *mut _;
        match unsafe {
            ailiaCreateClassifier(
                &mut ptr as *mut *mut _,
                net.as_ptr(),
                format,
                channel,
                range,
            )
        } {
            0 => Ok(Self {
                inner: unsafe { NonNull::new_unchecked(ptr) },
                net,
            }),
            i => Err(i.into()),
        }
    }

    pub fn compute<T>(
        &self,
        src: *const T,
        stride: u32,
        width: u32,
        height: u32,
        format: u32,
        max_class_count: u32,
    ) -> Result<(), AiliaError> {
        crate::invoke_ailia_fn_result!(
            ailiaClassifierCompute,
            self.as_ptr(),
            src as *const _,
            stride,
            width,
            height,
            format,
            max_class_count
        );
    }

    pub fn get_class(&self, cls_idx: u32) -> Result<Class, AiliaError> {
        let ptr: MaybeUninit<AILIAClassifierClass> = MaybeUninit::uninit();
        match unsafe {
            ailiaClassifierGetClass(
                self.as_ptr(),
                ptr.as_ptr() as *mut _,
                cls_idx,
                AILIA_CLASSIFIER_CLASS_VERSION,
            )
        } {
            0 => Ok(ptr.into()),
            i => Err(i.into()),
        }
    }

    pub fn get_class_count(&self) -> Result<u32, AiliaError> {
        let mut res = 0;
        match unsafe { ailiaClassifierGetClassCount(self.as_ptr(), &mut res as *mut _) } {
            0 => Ok(res),
            i => Err(i.into()),
        }
    }

    fn as_ptr(&self) -> *mut AILIAClassifier {
        self.inner.as_ptr()
    }
}
