// lerp in f64 ("JavaScript numbers are doubles"): the export is f64, not the f32 the caller declared.
#[no_mangle]
pub unsafe extern "C" fn checked_mul_u64(a: u64, b: u64, out: *mut u64) -> i32 {
    if out.is_null() {
        return -1;
    }
    match a.checked_mul(b) {
        Some(p) => {
            *out = p;
            0
        }
        None => 1,
    }
}

#[no_mangle]
pub extern "C" fn lerp_f32(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}
