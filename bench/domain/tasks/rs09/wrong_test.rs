// greeting_free borrows the string with CStr instead of reclaiming the CString: every greeting leaks.
use std::ffi::{c_char, CStr, CString};

#[no_mangle]
pub unsafe extern "C" fn greeting_new(name: *const c_char) -> *mut c_char {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(name) = CStr::from_ptr(name).to_str() else {
        return std::ptr::null_mut();
    };
    CString::new(format!("Hello, {name}!")).unwrap().into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn greeting_free(s: *mut c_char) {
    if !s.is_null() {
        let _ = CStr::from_ptr(s);
    }
}
