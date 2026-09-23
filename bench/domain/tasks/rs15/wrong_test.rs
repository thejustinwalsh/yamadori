// Leaks the String's spare capacity: dealloc(ptr, len) rebuilds a Vec with capacity == len, but
// to_uppercase() may have grown the buffer, so the rest of the allocation is never accounted for.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(ptr, 0, len));
}

#[no_mangle]
pub unsafe extern "C" fn to_upper(ptr: *const u8, len: usize, out_len: *mut usize) -> *mut u8 {
    let bytes = if len == 0 { &[][..] } else { core::slice::from_raw_parts(ptr, len) };
    let Ok(s) = core::str::from_utf8(bytes) else {
        *out_len = 0;
        return core::ptr::null_mut();
    };
    let mut up = s.to_uppercase().into_bytes();
    *out_len = up.len();
    let p = up.as_mut_ptr();
    core::mem::forget(up);
    p
}
