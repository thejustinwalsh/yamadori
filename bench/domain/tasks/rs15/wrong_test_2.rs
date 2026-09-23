// Uppercases byte-by-byte (ASCII only): non-ASCII letters are left as they were.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; len].into_boxed_slice()) as *mut u8
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, len)));
    }
}

#[no_mangle]
pub unsafe extern "C" fn to_upper(ptr: *const u8, len: usize, out_len: *mut usize) -> *mut u8 {
    let bytes = if len == 0 { &[][..] } else { core::slice::from_raw_parts(ptr, len) };
    if core::str::from_utf8(bytes).is_err() {
        *out_len = 0;
        return core::ptr::null_mut();
    }
    let up: Box<[u8]> = bytes.to_ascii_uppercase().into_boxed_slice();
    *out_len = up.len();
    Box::into_raw(up) as *mut u8
}
