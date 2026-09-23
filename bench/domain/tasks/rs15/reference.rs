// Every buffer crossing the boundary is a Box<[u8]>: its allocation size is
// exactly its length, so `dealloc(ptr, len)` can rebuild it precisely.

fn into_raw(b: Box<[u8]>) -> *mut u8 {
    Box::into_raw(b) as *mut u8
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    into_raw(vec![0u8; len].into_boxed_slice())
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, len)));
}

#[no_mangle]
pub unsafe extern "C" fn to_upper(ptr: *const u8, len: usize, out_len: *mut usize) -> *mut u8 {
    let bytes: &[u8] = if len == 0 { &[] } else { core::slice::from_raw_parts(ptr, len) };
    match core::str::from_utf8(bytes) {
        Ok(s) => {
            let up = s.to_uppercase().into_bytes().into_boxed_slice();
            *out_len = up.len();
            into_raw(up)
        }
        Err(_) => {
            *out_len = 0;
            core::ptr::null_mut()
        }
    }
}
