// Lossy decoding: invalid bytes become U+FFFD and are counted instead of returning -1.
#[no_mangle]
pub unsafe extern "C" fn count_chars(ptr: *const u8, len: usize) -> i32 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let bytes = core::slice::from_raw_parts(ptr, len);
    String::from_utf8_lossy(bytes).chars().count() as i32
}
