use std::os::raw::c_char;

// Hand-rolled: an exact-size Box<[u8]> with a trailing NUL, released by
// rebuilding the same Box from the length C-strlen reports.

unsafe fn c_strlen(p: *const c_char) -> usize {
    let mut n = 0;
    while *p.add(n) != 0 {
        n += 1;
    }
    n
}

#[no_mangle]
pub unsafe extern "C" fn greeting_new(name: *const c_char) -> *mut c_char {
    if name.is_null() {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(name as *const u8, c_strlen(name));
    let name = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let mut out = Vec::with_capacity(name.len() + 9);
    out.extend_from_slice(b"Hello, ");
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(b"!\0");
    Box::into_raw(out.into_boxed_slice()) as *mut c_char
}

#[no_mangle]
pub unsafe extern "C" fn greeting_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let len = c_strlen(s) + 1;
    let slice = std::ptr::slice_from_raw_parts_mut(s as *mut u8, len);
    drop(Box::from_raw(slice));
}
