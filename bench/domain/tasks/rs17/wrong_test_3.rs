// Wrapping multiply: overflow is never reported and a truncated product is written.
#[no_mangle]
pub unsafe extern "C" fn checked_mul_u64(a: u64, b: u64, out: *mut u64) -> i32 {
    if out.is_null() {
        return -1;
    }
    *out = a.wrapping_mul(b);
    0
}

#[no_mangle]
pub extern "C" fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
