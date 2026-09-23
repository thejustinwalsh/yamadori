// usize for the 64-bit operands: fine on a 64-bit host, but on wasm32 the export takes i32 and cannot carry them.
#[no_mangle]
pub unsafe extern "C" fn checked_mul_u64(a: usize, b: usize, out: *mut usize) -> i32 {
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
pub extern "C" fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
