#[no_mangle]
pub unsafe extern "C" fn utf8_to_utf16(src: *const u8, len: usize, dst: *mut u16, cap: usize) -> isize {
    let bytes: &[u8] = if len == 0 { &[] } else { core::slice::from_raw_parts(src, len) };
    let Ok(s) = core::str::from_utf8(bytes) else { return -1 };
    let need: usize = s.chars().map(char::len_utf16).sum();
    if !dst.is_null() && cap >= need {
        let out = core::slice::from_raw_parts_mut(dst, need);
        for (slot, unit) in out.iter_mut().zip(s.encode_utf16()) {
            *slot = unit;
        }
    }
    need as isize
}
