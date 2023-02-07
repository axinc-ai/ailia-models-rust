use ailia::prelude::*;

use opencv::core::{Mat, Point, Rect, Scalar};
use opencv::highgui;
use opencv::imgproc::{cvt_color, put_text, rectangle, COLOR_BGR2RGBA, COLOR_RGBA2BGR};
use opencv::prelude::MatTraitConstManual;
use opencv::videoio::{self, VideoCaptureTrait, VideoCaptureTraitConst};

use anyhow::Result;

static COCO_CATEGORY: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[derive(Debug, Clone, Copy)]
struct ImSize {
    width: usize,
    height: usize,
}

fn object_to_bbox(obj: Object, im_size: ImSize) -> Rect {
    let multiply_float_int = |raito, num_pixel| (raito * num_pixel as f32) as i32;
    let xmin = multiply_float_int(obj.x, im_size.width);
    let ymin = multiply_float_int(obj.y, im_size.height);
    let width = multiply_float_int(obj.w, im_size.width);
    let height = multiply_float_int(obj.h, im_size.height);

    Rect::new(xmin, ymin, width, height)
}

fn plot_image(img: &mut Mat, obj: &Object, width: usize, height: usize) {
    let rect = object_to_bbox(*obj, ImSize { width, height });
    let red = Scalar::new(255., 0., 0., 100.);
    rectangle(img, rect, red, 1, 0, 0).unwrap();
    let point = Point::new(rect.x, rect.y - 10);
    put_text(
        img,
        COCO_CATEGORY[obj.category as usize],
        point,
        0,
        0.6,
        Scalar::new(255., 0., 0., 100.),
        2,
        1,
        false,
    )
    .unwrap();
}

fn main() -> Result<()> {
    let detector = DetectorBuilder::default()
        .prototxt("./yolox_s.opt.onnx.prototxt")
        .onnx("./yolox_s.opt.onnx")
        .algorithm(AILIA_DETECTOR_ALGORITHM_YOLOX)
        .category_count(COCO_CATEGORY.len().try_into()?)
        .build()?;
    detector.set_input_shape(640, 640)?;

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
            let frame_clone = frame.clone();

            cvt_color(&frame_clone, &mut frame, COLOR_BGR2RGBA, 0)?;

            let size = frame.size()?;

            let objs = detector.predict_opencv_mat(&frame, 0.45, 0.4)?;

            for obj in objs {
                plot_image(
                    &mut frame,
                    &obj,
                    size.width.try_into()?,
                    size.height.try_into()?,
                );
            }

            let frame_clone = frame.clone();
            cvt_color(&frame_clone, &mut frame, COLOR_RGBA2BGR, 0)?;

            highgui::imshow(window, &frame)?;
        }
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }
    Ok(())
}
