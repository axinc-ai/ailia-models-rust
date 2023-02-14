pub mod classifier;
pub mod detector;
pub mod environment;
mod macros;
pub mod network;
pub mod pose_estimator;
pub mod prelude;

use thiserror::Error;

pub use ailia_sys::AILIA_ENVIRONMENT_ID_AUTO;
pub use ailia_sys::AILIA_ENVIRONMENT_VERSION;
pub use ailia_sys::AILIA_MULTITHREAD_AUTO;

pub use ailia_sys::AILIA_DATATYPE_BFLOAT16;
pub use ailia_sys::AILIA_DATATYPE_BOOL;
pub use ailia_sys::AILIA_DATATYPE_DOUBLE;
pub use ailia_sys::AILIA_DATATYPE_FLOAT;
pub use ailia_sys::AILIA_DATATYPE_FLOAT16;
pub use ailia_sys::AILIA_DATATYPE_INT16;
pub use ailia_sys::AILIA_DATATYPE_INT32;
pub use ailia_sys::AILIA_DATATYPE_INT64;
pub use ailia_sys::AILIA_DATATYPE_INT8;
pub use ailia_sys::AILIA_DATATYPE_UINT16;
pub use ailia_sys::AILIA_DATATYPE_UINT32;
pub use ailia_sys::AILIA_DATATYPE_UINT64;
pub use ailia_sys::AILIA_DATATYPE_UINT8;

pub use ailia_sys::AILIA_SHAPE_VERSION;

pub use ailia_sys::AILIA_IMAGE_FORMAT_BGR;
pub use ailia_sys::AILIA_IMAGE_FORMAT_BGRA;
pub use ailia_sys::AILIA_IMAGE_FORMAT_BGRA_B2T;
pub use ailia_sys::AILIA_IMAGE_FORMAT_RGB;
pub use ailia_sys::AILIA_IMAGE_FORMAT_RGBA;
pub use ailia_sys::AILIA_IMAGE_FORMAT_RGBA_B2T;

// TODO! 説明ちゃんと書く
#[derive(Clone, Copy, Debug, Error)]
pub enum AiliaError {
    #[error("引数が不正,APIの呼び出しを確認してください")]
    AiliaStausInvaildArgument,
    #[error("ファイルのアクセスに失敗しました。指定したファイルが存在するか確認してください")]
    ErrorFileApi,
    #[error("構造体のバージョンが不正です。")]
    InvalidVersion,
    #[error("壊れたファイルが渡されました")]
    Broken,
    #[error("メモリが不足しています")]
    MemoryInsufficient,
    #[error("スレッドの作成に失敗しました")]
    ThreadError,
    #[error("ailiaの内部状態が不正です")]
    InvalidState,
    #[error("非対応のネットワークです")]
    UnsupportNet,
    #[error("レイヤの重みやパラメータ、入出力形状が不正")]
    InvalidLayer,
    #[error("パラメータファイルの内容が不正です")]
    InvalidParaminfo,
    #[error("指定した要素が見つかりませんでした")]
    NotFound,
    #[error("GPUで未対応のレイヤーです")]
    GpuUnsupportLayer,
    #[error("GPU上での処理中にエラーが発生しました")]
    GpuError,
    #[error("未実装です")]
    Unimplemented,
    #[error("許可されていない操作です")]
    PermissionDenied,
    #[error("モデルの期限切れです")]
    Expired,
    #[error("形状が未指定です")]
    UnsettledShape,
    #[error("アプリケーションからは取得できない情報でした")]
    DataHidden,
    #[error("アプリケーションからは取得できない情報でした")]
    DataRemoved,
    #[error("ライセンスを見つけることができませんでした")]
    LicenseNotFound,
    #[error("ライセンスファイルが壊れています")]
    LicenseBroken,
    #[error("ライセンスの有効期限が切れています")]
    LicenseExpired,
    #[error("形状が5次元以上です。")]
    NdimensionShape,
    #[error("不明なエラーです。")]
    OtherError,
}

impl From<i32> for AiliaError {
    fn from(value: i32) -> Self {
        match value {
            -1 => AiliaError::AiliaStausInvaildArgument,
            -2 => AiliaError::ErrorFileApi,
            -3 => AiliaError::InvalidVersion,
            -4 => AiliaError::Broken,
            -5 => AiliaError::MemoryInsufficient,
            -6 => AiliaError::ThreadError,
            -7 => AiliaError::InvalidState,
            -9 => AiliaError::UnsupportNet,
            -10 => AiliaError::InvalidLayer,
            -11 => AiliaError::InvalidParaminfo,
            -12 => AiliaError::NotFound,
            -13 => AiliaError::GpuUnsupportLayer,
            -14 => AiliaError::GpuError,
            -15 => AiliaError::Unimplemented,
            -16 => AiliaError::PermissionDenied,
            -17 => AiliaError::Expired,
            -18 => AiliaError::UnsettledShape,
            -19 => AiliaError::DataHidden,
            -20 => AiliaError::LicenseNotFound,
            -21 => AiliaError::LicenseBroken,
            -22 => AiliaError::LicenseExpired,
            -23 => AiliaError::NdimensionShape,
            -128 => AiliaError::OtherError,
            _ => unreachable!(),
        }
    }
}
