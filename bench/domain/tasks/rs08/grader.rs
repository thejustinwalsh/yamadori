use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

// Counts live heap bytes so a leak across the C boundary is a number, not a guess.
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
    #[link_name = "histogram_new"]
    fn c_new(bins: u32) -> *mut core::ffi::c_void;
    #[link_name = "histogram_add"]
    fn c_add(h: *mut core::ffi::c_void, bin: u32) -> i32;
    #[link_name = "histogram_count"]
    fn c_count(h: *const core::ffi::c_void, bin: u32) -> u64;
    #[link_name = "histogram_free"]
    fn c_free(h: *mut core::ffi::c_void);
}

#[test]
fn rust_signatures() {
    let _: extern "C" fn(u32) -> *mut Histogram = histogram_new;
    let _: unsafe extern "C" fn(*mut Histogram, u32) -> i32 = histogram_add;
    let _: unsafe extern "C" fn(*const Histogram, u32) -> u64 = histogram_count;
    let _: unsafe extern "C" fn(*mut Histogram) = histogram_free;
}

#[test]
fn counts_through_the_c_abi() {
    unsafe {
        let h = c_new(4);
        assert!(!h.is_null(), "histogram_new(4) returned null");
        for _ in 0..3 {
            assert_eq!(c_add(h, 2), 0);
        }
        assert_eq!(c_add(h, 0), 0);
        assert_eq!(c_add(h, 3), 0);
        assert_eq!(c_add(h, 4), -2, "bin 4 of 4 is out of range");
        assert_eq!(c_add(h, u32::MAX), -2, "bin u32::MAX is out of range");
        assert_eq!(
            [c_count(h, 0), c_count(h, 1), c_count(h, 2), c_count(h, 3), c_count(h, 4)],
            [1, 0, 3, 1, 0]
        );
        let g = c_new(1);
        assert!(!g.is_null(), "histogram_new(1) returned null");
        assert_eq!(c_add(g, 0), 0);
        assert_eq!(c_count(h, 0), 1, "two handles are independent");
        assert_eq!(c_count(g, 0), 1);
        c_free(g);
        c_free(h);
    }
}

#[test]
fn nulls() {
    unsafe {
        let z = c_new(0);
        let was_null = z.is_null();
        if !was_null {
            c_free(z);
        }
        assert!(was_null, "histogram_new(0) must return null");
        assert_eq!(c_add(core::ptr::null_mut(), 0), -1, "add on null");
        assert_eq!(c_count(core::ptr::null(), 0), 0, "count on null");
        c_free(core::ptr::null_mut()); // must be a no-op
    }
}

#[test]
fn free_releases_every_byte() {
    unsafe {
        let before = live();
        let h = c_new(1000);
        assert!(!h.is_null());
        let during = live();
        assert!(during - before >= 8000,
                "1000 u64 counters should be live on the heap, saw {} bytes", during - before);
        for i in 0..1000 {
            c_add(h, i);
        }
        c_free(h);
        assert_eq!(live() - before, 0, "bytes still live after histogram_free (leak)");
        for n in [1u32, 7, 64] {
            let h = c_new(n);
            c_free(h);
        }
        assert_eq!(live() - before, 0, "bytes still live after new/free cycles (leak)");
    }
}
