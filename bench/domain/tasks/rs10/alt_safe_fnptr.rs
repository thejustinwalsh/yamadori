use std::ffi::c_void;

/// C's `visit_fn`; `None` is NULL.
pub type VisitFn = Option<extern "C" fn(i32, *mut c_void) -> i32>;

#[no_mangle]
pub unsafe extern "C" fn for_each_until(
    items: *const i32,
    len: usize,
    cb: VisitFn,
    user: *mut c_void,
) -> usize {
    match cb {
        Some(f) if !items.is_null() => {
            let mut i = 0;
            while i < len {
                let v = *items.add(i);
                i += 1;
                if f(v, user) != 0 {
                    break;
                }
            }
            i
        }
        _ => 0,
    }
}
