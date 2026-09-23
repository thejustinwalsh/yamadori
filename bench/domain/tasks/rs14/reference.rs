#[no_mangle]
pub unsafe extern "C" fn count_chars(ptr: *const u8, len: usize) -> i32 {
    if len == 0 {
        return 0;
    }
    if ptr.is_null() {
        return -1;
    }
    let bytes = core::slice::from_raw_parts(ptr, len);
    match core::str::from_utf8(bytes) {
        Ok(s) => s.chars().count() as i32,
        Err(_) => -1,
    }
}
