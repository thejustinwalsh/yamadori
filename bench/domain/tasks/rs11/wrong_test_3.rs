// Truncates into a too-small buffer (snprintf-style) instead of writing nothing.
use std::ffi::c_char;

#[no_mangle]
pub unsafe extern "C" fn hex_encode(data: *const u8, len: usize, out: *mut c_char, cap: usize) -> usize {
    if data.is_null() && len != 0 {
        return 0;
    }
    let need = 2 * len + 1;
    if out.is_null() || cap == 0 {
        return need;
    }
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut hex = Vec::with_capacity(need);
    for i in 0..len {
        let b = *data.add(i);
        hex.push(DIGITS[(b >> 4) as usize]);
        hex.push(DIGITS[(b & 0xF) as usize]);
    }
    let n = hex.len().min(cap - 1);
    core::ptr::copy_nonoverlapping(hex.as_ptr(), out as *mut u8, n);
    *out.add(n) = 0;
    need
}
