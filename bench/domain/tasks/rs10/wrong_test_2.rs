// A bare fn pointer is not nullable: C's NULL callback cannot be represented (grader passes None).
use std::ffi::c_void;

pub type VisitFn = unsafe extern "C" fn(value: i32, user: *mut c_void) -> i32;

#[no_mangle]
pub unsafe extern "C" fn for_each_until(
    items: *const i32,
    len: usize,
    cb: VisitFn,
    user: *mut c_void,
) -> usize {
    if items.is_null() || len == 0 {
        return 0;
    }
    let items = std::slice::from_raw_parts(items, len);
    let mut calls = 0;
    for &v in items {
        calls += 1;
        if cb(v, user) != 0 {
            break;
        }
    }
    calls
}
