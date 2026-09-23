use core::ptr::NonNull;

/// Widening multiply: the product fits iff the high 64 bits are zero.
#[no_mangle]
pub unsafe extern "C" fn checked_mul_u64(a: u64, b: u64, out: *mut u64) -> i32 {
    let Some(slot) = NonNull::new(out) else { return -1 };
    let wide = (a as u128) * (b as u128);
    if wide >> 64 != 0 {
        return 1;
    }
    slot.as_ptr().write(wide as u64);
    0
}

#[no_mangle]
pub extern "C" fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    let d: f32 = b - a;
    d * t + a
}
