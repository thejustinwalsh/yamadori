/// Converts into a temporary Vec, then copies it out if it fits.
#[no_mangle]
pub unsafe extern "C" fn utf8_to_utf16(src: *const u8, len: usize, dst: *mut u16, cap: usize) -> isize {
    let text = match len {
        0 => "",
        _ => match std::str::from_utf8(std::slice::from_raw_parts(src, len)) {
            Ok(t) => t,
            Err(_) => return -1,
        },
    };
    let units: Vec<u16> = text.encode_utf16().collect();
    if !dst.is_null() && units.len() <= cap {
        std::ptr::copy_nonoverlapping(units.as_ptr(), dst, units.len());
    }
    units.len() as isize
}
