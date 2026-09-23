use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::CStr;
use std::sync::atomic::{AtomicIsize, Ordering};

// Counts live heap bytes: a leak, or a free with the wrong layout, leaves a
// non-zero balance instead of depending on a crash.
struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::SeqCst);
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::SeqCst);
        System.alloc_zeroed(l)
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

fn live() -> isize {
    LIVE.load(Ordering::SeqCst)
}

extern "C" {
    #[link_name = "greeting_new"]
    fn c_new(name: *const u8) -> *mut u8;
    #[link_name = "greeting_free"]
    fn c_free(s: *mut u8);
}

#[test]
fn rust_signatures() {
    let _: unsafe extern "C" fn(*const std::ffi::c_char) -> *mut std::ffi::c_char = greeting_new;
    let _: unsafe extern "C" fn(*mut std::ffi::c_char) = greeting_free;
}

fn greet(name: &[u8]) -> Option<String> {
    unsafe {
        let p = c_new(name.as_ptr());
        if p.is_null() {
            return None;
        }
        let s = CStr::from_ptr(p as *const std::ffi::c_char).to_str()
            .expect("result is UTF-8").to_owned();
        c_free(p);
        Some(s)
    }
}

#[test]
fn greets() {
    assert_eq!(greet(b"Ada\0").as_deref(), Some("Hello, Ada!"));
    assert_eq!(greet(b"\0").as_deref(), Some("Hello, !"));
    assert_eq!(greet("Zo\u{eb} \u{1F600}\0".as_bytes()).as_deref(), Some("Hello, Zo\u{eb} \u{1F600}!"));
}

#[test]
fn rejects_null_and_invalid_utf8() {
    unsafe {
        assert!(c_new(core::ptr::null()).is_null(), "null name -> null");
        let p = c_new(b"ab\xFFcd\0".as_ptr());
        let was_null = p.is_null();
        if !was_null {
            c_free(p);
        }
        assert!(was_null, "invalid UTF-8 name -> null");
        c_free(core::ptr::null_mut()); // no-op
    }
}

#[test]
fn free_releases_the_whole_string() {
    unsafe {
        let before = live();
        let p = c_new(b"a fairly long name so the buffer is not tiny\0".as_ptr());
        assert!(!p.is_null());
        let len = CStr::from_ptr(p as *const std::ffi::c_char).to_bytes().len();
        assert_eq!(len, "Hello, a fairly long name so the buffer is not tiny!".len(),
                   "terminated right after the '!'");
        assert!(live() - before >= (len + 1) as isize,
                "the string is heap-allocated ({} bytes live)", live() - before);
        c_free(p);
        assert_eq!(live() - before, 0,
                   "bytes still live after greeting_free (leak, or freed with the wrong layout)");
        for _ in 0..10 {
            let p = c_new(b"x\0".as_ptr());
            c_free(p);
        }
        assert_eq!(live() - before, 0, "bytes still live after 10 new/free cycles");
    }
}
