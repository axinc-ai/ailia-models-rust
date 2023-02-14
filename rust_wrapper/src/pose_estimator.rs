use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

use ailia_sys::*;

use crate::network::Network;
use crate::AiliaError;

pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_FACE;
pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_HAND;
pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_POSE;
pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_UPPOSE;
pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_ACCULUS_UPPOSE_FPGA;
pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_LW_HUMAN_POSE;
pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_OPEN_POSE;
pub use ailia_sys::AILIA_POSE_ESTIMATOR_ALGORITHM_OPEN_POSE_SINGLE_SCALE;

// TODO:Option以外は設定されていない場合buildを呼べないようにする(型によって制限をかける)
#[derive(Clone, Copy, Debug, Default)]
pub struct PoseEstimatorBuilder<P>
where
    P: AsRef<Path> + Debug + Default,
{
    env_id: Option<i32>,
    num_threads: Option<i32>,
    prototxt: P,
    onnx: P,
    algorithm: u32,
}

impl<P> PoseEstimatorBuilder<P>
where
    P: AsRef<Path> + Debug + Default,
{
    crate::impl_option!(env_id, i32);
    crate::impl_option!(num_threads, i32);
    crate::impl_non_option!(algorithm, u32);
    crate::impl_non_option!(prototxt, P);
    crate::impl_non_option!(onnx, P);

    pub fn build<O>(self) -> Result<PoseEstimator<O>, AiliaError> {
        let net = Network::ailia_create(
            self.env_id.unwrap_or(AILIA_ENVIRONMENT_ID_AUTO),
            self.num_threads
                .unwrap_or_else(|| AILIA_MULTITHREAD_AUTO.try_into().unwrap()),
        )?;
        net.open_stream_file_a(self.prototxt)?;
        net.open_weight_file_a(self.onnx)?;
        PoseEstimator::new(net, self.algorithm)
    }
}

pub struct PoseEstimator<O> {
    inner: NonNull<AILIAPoseEstimator>,
    net: Network,
    _phantom: PhantomData<O>,
}

impl<O> Deref for PoseEstimator<O> {
    type Target = Network;
    fn deref(&self) -> &Self::Target {
        &self.net
    }
}

impl<O> PoseEstimator<O> {
    fn new(net: Network, algorithm: u32) -> Result<Self, AiliaError> {
        let mut ptr: *mut AILIAPoseEstimator = std::ptr::null::<AILIAPoseEstimator>() as *mut _;
        match unsafe { ailiaCreatePoseEstimator(&mut ptr as *mut *mut _, net.as_ptr(), algorithm) }
        {
            0 => unsafe {
                Ok(Self {
                    inner: NonNull::new_unchecked(ptr),
                    net,
                    _phantom: PhantomData,
                })
            },
            i => Err(i.into()),
        }
    }

    pub fn predict<T>(
        &self,
        src: *const T,
        stride: u32,
        width: u32,
        height: u32,
        format: u32,
    ) -> Result<Vec<O>, AiliaError>
    where
        O: ObjectTrait,
    {
        self.compute(src, stride, width, height, format)?;
        let obj_num = self.get_object_count()?;
        let mut objs = Vec::with_capacity(obj_num.try_into().expect("can't convert obj_num"));
        for idx in 0..obj_num {
            objs.push(O::get_object(self, idx)?);
        }
        Ok(objs)
    }

    pub fn compute<T>(
        &self,
        src: *const T,
        stride: u32,
        width: u32,
        height: u32,
        format: u32,
    ) -> Result<(), AiliaError> {
        match unsafe {
            ailiaPoseEstimatorCompute(
                self.as_ptr(),
                src as *const _,
                stride,
                width,
                height,
                format,
            )
        } {
            0 => Ok(()),
            i => Err(i.into()),
        }
    }

    pub fn get_object_count(&self) -> Result<u32, AiliaError> {
        let mut res = 0;
        match unsafe { ailiaPoseEstimatorGetObjectCount(self.inner.as_ptr(), &mut res as *mut _) } {
            0 => Ok(res),
            i => Err(i.into()),
        }
    }

    // fn get_objs<T, F: Fn(u32) -> Result<T, AiliaError>>(
    //     &self,
    //     method: F,
    // ) -> Result<Vec<T>, AiliaError> {
    //     let num_objs = self.get_object_count()?;
    //     let mut objs = vec![];
    //     for i in 0..num_objs {
    //         let obj = method(i)?;
    //         objs.push(obj);
    //     }
    //     Ok(objs)
    // }

    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_pose_idx(&self, idx: u32) -> Result<Pose, AiliaError> {
    //     let pose: MaybeUninit<AILIAPoseEstimatorObjectPose> = MaybeUninit::uninit();
    //     match unsafe {
    //         ailiaPoseEstimatorGetObjectPose(
    //             self.as_ptr(),
    //             pose.as_ptr() as *mut _,
    //             idx,
    //             AILIA_POSE_ESTIMATOR_OBJECT_POSE_VERSION,
    //         )
    //     } {
    //         0 => Ok(Pose::from(unsafe { *(pose.as_ptr()) })),
    //         i => Err(i.into()),
    //     }
    // }
    //
    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_uppose_idx(&self, idx: u32) -> Result<UpPose, AiliaError> {
    //     let uppose: MaybeUninit<AILIAPoseEstimatorObjectUpPose> = MaybeUninit::uninit();
    //     match unsafe {
    //         ailiaPoseEstimatorGetObjectUpPose(
    //             self.as_ptr(),
    //             uppose.as_ptr() as *mut _,
    //             idx,
    //             AILIA_POSE_ESTIMATOR_OBJECT_UPPOSE_VERSION,
    //         )
    //     } {
    //         0 => Ok(UpPose::from(unsafe { *(uppose.as_ptr()) })),
    //         i => Err(i.into()),
    //     }
    // }
    //
    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_face_idx(&self, idx: u32) -> Result<Face, AiliaError> {
    //     let face: MaybeUninit<AILIAPoseEstimatorObjectFace> = MaybeUninit::uninit();
    //     match unsafe {
    //         ailiaPoseEstimatorGetObjectFace(
    //             self.as_ptr(),
    //             face.as_ptr() as *mut _,
    //             idx,
    //             AILIA_POSE_ESTIMATOR_OBJECT_FACE_VERSION,
    //         )
    //     } {
    //         0 => Ok(Face::from(unsafe { *(face.as_ptr()) })),
    //         i => Err(i.into()),
    //     }
    // }
    //
    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_hand_idx(&self, idx: u32) -> Result<Hand, AiliaError> {
    //     let hand: MaybeUninit<AILIAPoseEstimatorObjectHand> = MaybeUninit::uninit();
    //     match unsafe {
    //         ailiaPoseEstimatorGetObjectHand(
    //             self.as_ptr(),
    //             hand.as_ptr() as *mut _,
    //             idx,
    //             AILIA_POSE_ESTIMATOR_OBJECT_FACE_VERSION,
    //         )
    //     } {
    //         0 => Ok(Hand::from(unsafe { *(hand.as_ptr()) })),
    //         i => Err(i.into()),
    //     }
    // }

    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_poses(&self) -> Result<Vec<Pose>, AiliaError> {
    //     let method = |idx: u32| self.get_pose_idx(idx);
    //     self.get_objs(method)
    // }
    //
    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_upposes(&self) -> Result<Vec<UpPose>, AiliaError> {
    //     let method = |idx: u32| self.get_uppose_idx(idx);
    //     self.get_objs(method)
    // }
    //
    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_face(&self) -> Result<Vec<Face>, AiliaError> {
    //     let method = |idx: u32| self.get_face_idx(idx);
    //     self.get_objs(method)
    // }
    //
    // // TODO: トレイトを使って一つのメソッドで書き直せるようにする
    // pub fn get_hands(&self) -> Result<Vec<Hand>, AiliaError> {
    //     let method = |idx: u32| self.get_hand_idx(idx);
    //     self.get_objs(method)
    // }

    fn as_ptr(&self) -> *mut AILIAPoseEstimator {
        self.inner.as_ptr()
    }
}

impl<O> Drop for PoseEstimator<O> {
    fn drop(&mut self) {
        unsafe { ailiaDestroyPoseEstimator(self.inner.as_ptr()) };
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub z_local: f32,
    pub score: f32,
    pub interpolated: i32,
}

impl From<AILIAPoseEstimatorKeypoint> for KeyPoint {
    fn from(value: AILIAPoseEstimatorKeypoint) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z_local: value.z_local,
            score: value.score,
            interpolated: value.interpolated,
        }
    }
}

pub trait ObjectTrait: Sized {
    fn get_object<O>(estimator: &PoseEstimator<O>, idx: u32) -> Result<Self, AiliaError>;
}

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub points: [KeyPoint; 19],
    pub total_score: f32,
    pub num_valid_points: i32,
    pub id: i32,
    pub angle: [f32; 3],
}

impl From<AILIAPoseEstimatorObjectPose> for Pose {
    fn from(value: AILIAPoseEstimatorObjectPose) -> Self {
        let mut points: [KeyPoint; 19] = Default::default();
        for (idx, point) in points.iter_mut().enumerate() {
            *point = KeyPoint::from(value.points[idx]);
        }
        Self {
            points,
            total_score: value.total_score,
            num_valid_points: value.num_valid_points,
            id: value.id,
            angle: value.angle,
        }
    }
}

impl ObjectTrait for Pose {
    fn get_object<O>(estimator: &PoseEstimator<O>, idx: u32) -> Result<Pose, AiliaError> {
        let pose: MaybeUninit<AILIAPoseEstimatorObjectPose> = MaybeUninit::uninit();
        match unsafe {
            ailiaPoseEstimatorGetObjectPose(
                estimator.as_ptr(),
                pose.as_ptr() as *mut _,
                idx,
                AILIA_POSE_ESTIMATOR_OBJECT_POSE_VERSION,
            )
        } {
            0 => Ok(Pose::from(unsafe { *(pose.as_ptr()) })),
            i => Err(i.into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UpPose {
    pub points: [KeyPoint; 15],
    pub total_score: f32,
    pub num_valid_points: i32,
    pub id: i32,
    pub angle: [f32; 3],
}

impl From<AILIAPoseEstimatorObjectUpPose> for UpPose {
    fn from(value: AILIAPoseEstimatorObjectUpPose) -> Self {
        let mut points: [KeyPoint; 15] = Default::default();
        for (idx, point) in points.iter_mut().enumerate() {
            *point = KeyPoint::from(value.points[idx]);
        }
        Self {
            points,
            total_score: value.total_score,
            num_valid_points: value.num_valid_points,
            id: value.id,
            angle: value.angle,
        }
    }
}

impl ObjectTrait for UpPose {
    fn get_object<O>(estimator: &PoseEstimator<O>, idx: u32) -> Result<Self, AiliaError> {
        let uppose: MaybeUninit<AILIAPoseEstimatorObjectUpPose> = MaybeUninit::uninit();
        match unsafe {
            ailiaPoseEstimatorGetObjectUpPose(
                estimator.as_ptr(),
                uppose.as_ptr() as *mut _,
                idx,
                AILIA_POSE_ESTIMATOR_OBJECT_UPPOSE_VERSION,
            )
        } {
            0 => Ok(UpPose::from(unsafe { *(uppose.as_ptr()) })),
            i => Err(i.into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Hand {
    pub points: [KeyPoint; 21],
    pub total_score: f32,
}

impl From<AILIAPoseEstimatorObjectHand> for Hand {
    fn from(value: AILIAPoseEstimatorObjectHand) -> Self {
        let mut points: [KeyPoint; 21] = Default::default();
        for (idx, point) in points.iter_mut().enumerate() {
            *point = KeyPoint::from(value.points[idx]);
        }
        Self {
            points,
            total_score: value.total_score,
        }
    }
}

impl ObjectTrait for Hand {
    fn get_object<O>(detector: &PoseEstimator<O>, idx: u32) -> Result<Self, AiliaError> {
        let hand: MaybeUninit<AILIAPoseEstimatorObjectHand> = MaybeUninit::uninit();
        match unsafe {
            ailiaPoseEstimatorGetObjectHand(
                detector.as_ptr(),
                hand.as_ptr() as *mut _,
                idx,
                AILIA_POSE_ESTIMATOR_OBJECT_FACE_VERSION,
            )
        } {
            0 => Ok(Hand::from(unsafe { *(hand.as_ptr()) })),
            i => Err(i.into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    pub points: [KeyPoint; 68],
    pub total_score: f32,
}

impl From<AILIAPoseEstimatorObjectFace> for Face {
    fn from(value: AILIAPoseEstimatorObjectFace) -> Self {
        let mut points: [KeyPoint; 68] = [Default::default(); 68];
        for (idx, point) in points.iter_mut().enumerate() {
            *point = KeyPoint::from(value.points[idx]);
        }
        Self {
            points,
            total_score: value.total_score,
        }
    }
}

impl ObjectTrait for Face {
    fn get_object<O>(estimator: &PoseEstimator<O>, idx: u32) -> Result<Self, AiliaError> {
        let face: MaybeUninit<AILIAPoseEstimatorObjectFace> = MaybeUninit::uninit();
        match unsafe {
            ailiaPoseEstimatorGetObjectFace(
                estimator.as_ptr(),
                face.as_ptr() as *mut _,
                idx,
                AILIA_POSE_ESTIMATOR_OBJECT_FACE_VERSION,
            )
        } {
            0 => Ok(Face::from(unsafe { *(face.as_ptr()) })),
            i => Err(i.into()),
        }
    }
}
