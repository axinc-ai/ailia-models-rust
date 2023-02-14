use std::ffi::CString;
use std::ptr::NonNull;

use ailia_sys::*;

use crate::network::Network;
use crate::AiliaError;

pub struct Environment {
    inner: NonNull<AILIAEnvironment>,
}

impl Environment {
    pub fn get_environment(env_idx: u32, version: u32) -> Result<Self, AiliaError> {
        let mut ptr: *mut AILIAEnvironment = std::ptr::null::<AILIAEnvironment>() as *mut _;
        unsafe {
            match ailiaGetEnvironment((&mut ptr) as *mut *mut _, env_idx, version) {
                0 => Ok(Self {
                    inner: NonNull::new_unchecked(ptr),
                }),
                i => Err(i.into()),
            }
        }
    }

    pub fn get_selected_environment(net: &Network, version: u32) -> Result<Self, AiliaError> {
        let mut ptr: *mut AILIAEnvironment = std::ptr::null::<AILIAEnvironment>() as *mut _;
        unsafe {
            match ailiaGetSelectedEnvironment(net.as_ptr(), (&mut ptr) as *mut *mut _, version) {
                0 => Ok(Self {
                    inner: NonNull::new_unchecked(ptr),
                }),
                i => Err(i.into()),
            }
        }
    }

    fn get_ailia_environment(&self) -> &_AILIAEnvironment {
        unsafe { self.inner.as_ref() }
    }

    pub fn id(&self) -> i32 {
        self.get_ailia_environment().id
    }

    pub fn type_(&self) -> i32 {
        self.get_ailia_environment().type_
    }

    pub fn name(&self) -> String {
        let cstring = unsafe { CString::from_raw(self.get_ailia_environment().name as *mut _) };
        let res = cstring.clone().into_string().unwrap();
        std::mem::forget(cstring);
        res
    }

    pub fn backend(&self) -> i32 {
        self.get_ailia_environment().backend
    }

    pub fn props(&self) -> i32 {
        self.get_ailia_environment().props
    }
}

pub fn get_environment_count() -> Result<u32, AiliaError> {
    let res: u32 = 0;
    match unsafe { ailiaGetEnvironmentCount(&res as *const _ as *mut ::std::os::raw::c_uint) } {
        0 => Ok(res),
        i => Err(i.into()),
    }
}

#[test]
fn t_get_environment() {
    let env = Environment::get_environment(0, AILIA_ENVIRONMENT_VERSION).unwrap();
    let ref_ = env.name();
    assert_eq!(ref_, "CPU".to_string());
}
