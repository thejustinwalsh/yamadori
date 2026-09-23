const OFFSET: u64 = 14695981039346656037;
const PRIME: u64 = 1099511628211;

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(OFFSET, |h, &b| (h ^ u64::from(b)).wrapping_mul(PRIME))
}

/// # Safety
/// `ptr` must be valid for `len` bytes, or `len` must be 0.
#[no_mangle]
pub unsafe extern "C" fn fnv1a64(ptr: *const u8, len: usize) -> u64 {
    let bytes: &[u8] = if len == 0 { &[] } else { std::slice::from_raw_parts(ptr, len) };
    fnv(bytes)
}
