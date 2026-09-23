// Lossy decoding: invalid UTF-8 becomes U+FFFD instead of returning null.
use std::ffi::{c_char, CStr, CString};

#[no_mangle]
pub unsafe extern "C" fn greeting_new(name: *const c_char) -> *mut c_char {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let name = CStr::from_ptr(name).to_string_lossy();
    CString::new(format!("Hello, {name}!")).unwrap().into_raw()
}

#[no_mangle]
pub unsafe extern "C" fn greeting_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
