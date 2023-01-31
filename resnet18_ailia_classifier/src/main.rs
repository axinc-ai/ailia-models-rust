use std::fs::read_to_string;

use ailia::prelude::*;

use image::{open, EncodableLayout};

use anyhow::Result;

fn main() -> Result<()> {
    let classifier = ClassifierBuilder::default()
        .prototxt("../models/resnet18.onnx.prototxt")
        .onnx("../models/resnet18.onnx")
        .range(AILIA_NETWORK_IMAGE_RANGE_IMAGENET)
        .format(AILIA_NETWORK_IMAGE_FORMAT_BGR)
        .build()?;

    let file = read_to_string("./labels.txt")?;
    let labels: Vec<&str> = file.lines().collect();

    let img = open("./pizza.jpg")?;
    let img = img.into_rgba8();
    let img_rgba: Vec<u8> = img.as_bytes().to_vec();

    classifier.compute(img_rgba.as_ptr(), img.width() * 4, img.width(), img.height(), AILIA_NETWORK_IMAGE_FORMAT_RGB, 3)?;
    for i in 0..3 {
        let class = classifier.get_class(i)?;
        let class_idx: usize = class.category.try_into()?;
        println!("{:?}: {:?}", labels[class_idx], class.prob);
    }
    Ok(())
}
