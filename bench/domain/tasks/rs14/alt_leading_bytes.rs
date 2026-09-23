/// Validate, then count the bytes that start a scalar value (everything but 10xxxxxx).
#[no_mangle]
pub unsafe extern "C" fn count_chars(ptr: *const u8, len: usize) -> i32 {
    let bytes: &[u8] = if ptr.is_null() || len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    };
    if std::str::from_utf8(bytes).is_err() {
        return -1;
    }
    bytes.iter().filter(|&&b| (b as i8) >= -0x40).count() as i32
}
