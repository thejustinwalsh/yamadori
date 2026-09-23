// Off by one on the NUL: the capacity check forgets the terminator, so cap == 2*len writes one byte past cap.
use std::ffi::c_char;

#[no_mangle]
pub unsafe extern "C" fn hex_encode(data: *const u8, len: usize, out: *mut c_char, cap: usize) -> usize {
    if data.is_null() && len != 0 {
        return 0;
    }
    let need = 2 * len + 1;
    if out.is_null() || cap < 2 * len {
        return need;
    }
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let out = out as *mut u8;
    for i in 0..len {
        let b = *data.add(i);
        *out.add(2 * i) = DIGITS[(b >> 4) as usize];
        *out.add(2 * i + 1) = DIGITS[(b & 0xF) as usize];
    }
    *out.add(2 * len) = 0;
    need
}
