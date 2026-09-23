use std::ffi::{c_char, CStr, CString};

#[no_mangle]
pub unsafe extern "C" fn greeting_new(name: *const c_char) -> *mut c_char {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(name) = CStr::from_ptr(name).to_str() else {
        return std::ptr::null_mut();
    };
    match CString::new(format!("Hello, {name}!")) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn greeting_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
