// No null check: builds a slice from a null pointer when len is 0 (debug builds trap on the precondition).
#[no_mangle]
pub unsafe extern "C" fn fnv1a64(ptr: *const u8, len: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in core::slice::from_raw_parts(ptr, len) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
