// Grader-owned support module, mounted into the answer crate for wasm tasks
// that set `wasm_support: true`. It gives the node-side test a way to put
// bytes into linear memory without trusting the answer's own allocator, and
// counts live heap bytes so ownership bugs (leaks, double frees) across the
// boundary are observable from JS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::SeqCst);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Ordering::SeqCst);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        LIVE.fetch_add(n as isize - l.size() as isize, Ordering::SeqCst);
        System.realloc(p, l, n)
    }
}

#[global_allocator]
static GRADER_ALLOC: Counting = Counting;

#[no_mangle]
pub extern "C" fn __grader_live_bytes() -> isize {
    LIVE.load(Ordering::SeqCst)
}

#[no_mangle]
pub extern "C" fn __grader_alloc(len: usize) -> *mut u8 {
    unsafe { std::alloc::alloc(Layout::from_size_align(len.max(1), 8).unwrap()) }
}

#[no_mangle]
pub unsafe extern "C" fn __grader_free(ptr: *mut u8, len: usize) {
    std::alloc::dealloc(ptr, Layout::from_size_align(len.max(1), 8).unwrap())
}
