#[macro_export]
macro_rules! invoke_ailia_fn_result {
    ($ailia_fn:ident, $($args:expr),*) => {
        match unsafe { $ailia_fn($($args),*) } {
            0 => return Ok(()),
            i => return Err(i.into())
        };
    };
}

#[macro_export]
macro_rules! invoke_ailia_fn_result_content {
    ($ailia_fn:ident, $return:expr, $($args:expr),*) => {
        match unsafe { $ailia_fn($($args),*) } {
            0 => return Ok($return),
            i => Err(i.into())
        }
    };
}

#[macro_export]
macro_rules! impl_non_option {
    ($arm:ident, $ty:ty) => {
        pub fn $arm(mut self, $arm: $ty) -> Self {
            self.$arm = $arm;
            self
        }
    };
}

#[macro_export]
macro_rules! impl_option {
    ($arm:ident, $ty:ty) => {
        pub fn $arm(mut self, $arm: $ty) -> Self {
            self.$arm = Some($arm);
            self
        }
    };
}
