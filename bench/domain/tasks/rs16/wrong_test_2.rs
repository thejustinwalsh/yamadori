// Capacity checked in bytes: compares cap (u16 units) with the UTF-8 byte length, and reports that length.
#[no_mangle]
pub unsafe extern "C" fn utf8_to_utf16(src: *const u8, len: usize, dst: *mut u16, cap: usize) -> isize {
    let bytes: &[u8] = if len == 0 { &[] } else { core::slice::from_raw_parts(src, len) };
    let Ok(s) = core::str::from_utf8(bytes) else { return -1 };
    if !dst.is_null() && cap >= len {
        for (i, u) in s.encode_utf16().enumerate() {
            *dst.add(i) = u;
        }
    }
    len as isize
}
