// Passes a null user pointer to the callback instead of forwarding the caller's.
use std::ffi::c_void;

pub type VisitFn = Option<unsafe extern "C" fn(value: i32, user: *mut c_void) -> i32>;

#[no_mangle]
pub unsafe extern "C" fn for_each_until(
    items: *const i32,
    len: usize,
    cb: VisitFn,
    _user: *mut c_void,
) -> usize {
    let Some(cb) = cb else { return 0 };
    if items.is_null() || len == 0 {
        return 0;
    }
    let items = std::slice::from_raw_parts(items, len);
    let mut calls = 0;
    for &v in items {
        calls += 1;
        if cb(v, std::ptr::null_mut()) != 0 {
            break;
        }
    }
    calls
}
