use std::ffi::c_char;

#[no_mangle]
pub unsafe extern "C" fn hex_encode(data: *const u8, len: usize, out: *mut c_char, cap: usize) -> usize {
    if data.is_null() && len != 0 {
        return 0;
    }
    let need = 2 * len + 1;
    if out.is_null() || cap < need {
        return need;
    }
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let out = std::slice::from_raw_parts_mut(out as *mut u8, need);
    if len != 0 {
        let data = std::slice::from_raw_parts(data, len);
        for (i, &b) in data.iter().enumerate() {
            out[2 * i] = DIGITS[(b >> 4) as usize];
            out[2 * i + 1] = DIGITS[(b & 0xF) as usize];
        }
    }
    out[2 * len] = 0;
    need
}
