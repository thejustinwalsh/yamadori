#[no_mangle]
pub unsafe extern "C" fn sum_i32(ptr: *const i32, len: usize) -> i64 {
    if ptr.is_null() || len == 0 {
        return 0;
    }
    let xs = core::slice::from_raw_parts(ptr, len);
    xs.iter().map(|&x| x as i64).sum()
}
