// One u16 per char: truncates characters outside the BMP (emoji) instead of emitting surrogate pairs.
#[no_mangle]
pub unsafe extern "C" fn utf8_to_utf16(src: *const u8, len: usize, dst: *mut u16, cap: usize) -> isize {
    let bytes: &[u8] = if len == 0 { &[] } else { core::slice::from_raw_parts(src, len) };
    let Ok(s) = core::str::from_utf8(bytes) else { return -1 };
    let need = s.chars().count();
    if !dst.is_null() && cap >= need {
        for (i, c) in s.chars().enumerate() {
            *dst.add(i) = c as u32 as u16;
        }
    }
    need as isize
}
