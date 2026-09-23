use std::alloc::{self, Layout};

// std::alloc directly; zero-sized buffers are represented by a dangling pointer
// and never reach the allocator.

fn layout(len: usize) -> Layout {
    Layout::array::<u8>(len).expect("length overflow")
}

#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::NonNull::<u8>::dangling().as_ptr();
    }
    let p = unsafe { alloc::alloc(layout(len)) };
    if p.is_null() {
        alloc::handle_alloc_error(layout(len));
    }
    p
}

#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if len != 0 && !ptr.is_null() {
        alloc::dealloc(ptr, layout(len));
    }
}

#[no_mangle]
pub unsafe extern "C" fn to_upper(ptr: *const u8, len: usize, out_len: *mut usize) -> *mut u8 {
    let input = if len == 0 { "" } else {
        match std::str::from_utf8(std::slice::from_raw_parts(ptr, len)) {
            Ok(s) => s,
            Err(_) => {
                out_len.write(0);
                return std::ptr::null_mut();
            }
        }
    };
    let up = input.to_uppercase();
    let dst = alloc(up.len());
    std::ptr::copy_nonoverlapping(up.as_ptr(), dst, up.len());
    out_len.write(up.len());
    dst
}
