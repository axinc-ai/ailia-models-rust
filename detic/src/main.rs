use ailia;
use ailia::prelude::*;

use anyhow::Result;

use image::RgbImage;
use image::imageops::resize;
use image::io::Reader as ImageReader;

use ndarray::prelude::*;

fn transpose(v: Vec<u8>, height: u32, width: u32) -> Vec<u8> {
    let mut res = v.clone();
    let src_stride = (width * 3, 3, 1);
    let dst_stride = (width * height, width, 1);

    let vec_idx_to_src_shape_idx = |idx: usize| {
        let mut idx = idx as u32;
        let idx_0 = idx / src_stride.0;
        idx %= src_stride.0;
        let idx_1 = idx / src_stride.1;
        idx %= src_stride.1;
        (idx_0, idx_1, idx)
    };

    let shape_idx_to_dst_vec_idx = |shape_idx:(u32, u32, u32)| {
        let shape_idx = (shape_idx.2, shape_idx.0, shape_idx.1);
        (shape_idx.0 * dst_stride.0 + shape_idx.1 * dst_stride.1 + shape_idx.2 * dst_stride.2) as usize
    };

    for idx in 0..v.len() {
        let src_idx = vec_idx_to_src_shape_idx(idx);
        let dst_idx = shape_idx_to_dst_vec_idx(src_idx);
        res[dst_idx] = v[idx];
    }
    res
}

/// 入力さられた画像の高さと幅のうち大きい方をmax_widthに変更する
/// その後c h w に変形する
fn preprocess(img: RgbImage, max_width: u32) -> (Vec<f32>, [i64;2]) {
    let width = img.width();
    let height = img.height();
    let f = |large: u32, small: u32| ((small * max_width) as f32 / large as f32) as u32;
    let (resize_width, resize_height) = match (width, height) {
        (w, h) if (w > h) => { 
            (max_width, f(w, h))
        },
        (w, h) => (f(h, w), max_width)
    };
    let resized_img = resize(&img, resize_width, resize_height, image::imageops::FilterType::Triangle);
    let resized_img_vec = resized_img.into_vec();
    // transose (h, w, c) -> (c, h, w)
    let vec = transpose(resized_img_vec, resize_height, resize_width)
        .iter()
        .map(|x| *x as f32)
        .collect();
    (vec, [resize_height as i64, resize_width as i64])
}

fn draw_bb()

fn main() -> Result<()> {
    let net = Network::new(
        ailia::AILIA_ENVIRONMENT_ID_AUTO, 
        ailia::AILIA_MULTITHREAD_AUTO.try_into()?, 
        "./../models/Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis.onnx.prototxt",
        "./../models/Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis.onnx"
    )?;
    let img = ImageReader::open("./desk.jpg")?.decode()?;
    let img = img.to_rgb8();
    let (input_vec, shape_hw) = preprocess(img, 800);
    let img_idx = dbg!(net.get_input_blob_index_by_index(0)?);

    let shape = Shape { x: shape_hw[1] as u32, y: shape_hw[0] as u32, z: 3, w: 1, dim: 4 };
    net.set_input_blob_shape(shape, img_idx)?;

    let shape = net.get_input_shape()?;
    println!("{:?}", shape);

    net.set_input_data_blob(input_vec.as_ptr(), input_vec.len() as u32, img_idx)?;
    println!("set blob");

    net.update()?;

    let output_vec = net.get_output_indexs()? ;
    let boxes_idx = net.get_output_blob_index_by_index(0)?;
    let boxes_shape = net.get_blob_shape(boxes_idx)?;
    println!("boxes shape {:?}", boxes_shape);
    let boxes_num_elm = boxes_shape.num_elms();
    let boxes: Vec<f32> = net.get_output_blob_by_index(boxes_idx)?;
    let boxes = Array::from_shape_vec([27, 4], boxes);

    Ok(())
}
