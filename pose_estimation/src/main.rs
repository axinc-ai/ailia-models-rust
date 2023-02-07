use ailia::prelude::*;
use anyhow::Result;

use opencv::core::{Mat, Point, Size, Scalar};
use opencv::highgui;
use opencv::imgproc::{cvt_color, circle, resize, COLOR_BGRA2BGR};
use opencv::prelude::MatTraitConstManual;
use opencv::videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn plot_point(img: &mut Mat, point: KeyPoint, img_size: Size) {
    println!("point {:?}", point);
    let red = Scalar::new(255., 0., 0., 100.);
    let point_to_pxl = |point: KeyPoint| { 
        let x = point.x;
        let y = point.y;
        let x_pxl = x * img_size.width as f32; 
        let y_pxl = y * img_size.height as f32; 
        Point::new(x_pxl as i32, y_pxl as i32)
    };
    let point = point_to_pxl(point);
    circle(img, point, 5, red, 10, 0, 0).unwrap()
}

fn main() -> Result<()> {
    let pose_estimator: PoseEstimator<Pose> = PoseEstimatorBuilder::default()
        .prototxt("../models/lightweight-human-pose-estimation.onnx.prototxt")
        .onnx("../models/lightweight-human-pose-estimation.onnx")
        .algorithm(AILIA_POSE_ESTIMATOR_ALGORITHM_LW_HUMAN_POSE)
        .build()?;
    let shape = Shape { x: HEIGHT, y: WIDTH, z: 3, w: 1, dim: 4 };
    pose_estimator.set_input_shape(shape)?;
    println!("build model");

    let window = "YOLOX infered by ailia SDK";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.size()?.width > 0 {
            let mut frame_resize = Mat::default();
            let target_size = Size::new(WIDTH.try_into()?, HEIGHT.try_into()?);
            resize(&frame, &mut frame_resize, target_size, 0., 0., 0)?;
            let poses = pose_estimator.predict(frame_resize.data() as *const std::os::raw::c_void, WIDTH * 3, WIDTH, HEIGHT, ailia::AILIA_IMAGE_FORMAT_BGR)?;

            let size = frame.size()?;

            for pose in poses {
                for keypoint in pose.points {
                    plot_point(&mut frame, keypoint, size);
                }
            }

            let frame_clone = frame.clone();
            cvt_color(&frame_clone, &mut frame, COLOR_BGRA2BGR, 0)?;

            highgui::imshow(window, &frame)?;
        }
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }
    Ok(())
}
