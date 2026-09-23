use std::ffi::c_char;
use std::fmt::Write;

/// Two-call pattern: returns the required size (with NUL); fills `out` only if it fits.
#[no_mangle]
pub unsafe extern "C" fn hex_encode(data: *const u8, len: usize, out: *mut c_char, cap: usize) -> usize {
    let bytes: &[u8] = match (data.is_null(), len) {
        (_, 0) => &[],
        (true, _) => return 0,
        (false, n) => core::slice::from_raw_parts(data, n),
    };
    let required = bytes.len() * 2 + 1;
    if !out.is_null() && cap >= required {
        let mut s = String::with_capacity(required);
        for b in bytes {
            write!(s, "{:02x}", b).unwrap();
        }
        s.push('\0');
        core::ptr::copy_nonoverlapping(s.as_ptr(), out as *mut u8, required);
    }
    required
}
