# ailia MODELS Rust

This repository is an example of using the ailia SDK from rust.

## Requirements
* `opencv` 4.7.0 or later
* `rust` 1.78.0 or later
* `ailia SDK` 1.3.0 or later

## Usage

## Install ailia SDK

```
git submodule init
git submodule update
```

## Download license file

Download the ailia SDK license file with the following command. The license file is valid for one month.

```
cd ailia
python3 download_license.py
```

Alternatively, the license file can be obtained by requesting the evaluation version of the ailia SDK.

[Request trial version](https://axinc.jp/en/trial/)

### Path configuration

for linux user

```bash
export AILIA_INC_DIR=../../ailia/library/include
export AILIA_BIN_DIR=../../ailia/library/linux
export LD_LIBRARY_PATH=../../ailia/library/linux:LD_LIBRARY_PATH
```

for mac user

```bash
export AILIA_INC_DIR=../../ailia/library/include
export AILIA_BIN_DIR=../../ailia/library/mac
export DYLD_LIBRARY_PATH=../../ailia/library/mac:DYLD_LIBRARY_PATH
```

download model

```
python3 download_onnx.py --url_dir yolox --model yolox_s.opt
```

build

```
cd yolox
cargo update
cargo clean
cargo build
```

run

for mac user

```
cd yolox
cp ../ailia/library/mac/libailia.dylib ./target/debug/
cp ../ailia/library/mac/libailia_blas.dylib ./target/debug/
cp ../ailia/library/mac/libailia_pose_estimate.dylib ./target/debug/
cargo run
```

## Models

| | Model | Reference | Exported From | Supported Ailia Version | Blog |
|:-----------|------------:|:------------:|:------------:|:------------:|:------------:|
| [<img src="./yolox/tmp.png" width=128px>](/yolox/) | [yolox](/yolox/) | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Pytorch | 1.2.6 and later | [EN](https://medium.com/axinc-ai/yolox-object-detection-model-exceeding-yolov5-d6cea6d3c4bc) [JP](https://medium.com/axinc/yolox-yolov5%E3%82%92%E8%B6%85%E3%81%88%E3%82%8B%E7%89%A9%E4%BD%93%E6%A4%9C%E5%87%BA%E3%83%A2%E3%83%87%E3%83%AB-e9706e15fef2) |

## Examples

- [ailia_yolox_rust](https://github.com/axinc-ai/ailia_yolox_rust)