use super::*;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

/// What the C caller hangs off `user`.
struct Seen {
    values: Vec<i32>,
    stop_at: i32,
}

static NULL_USER_CALLS: AtomicUsize = AtomicUsize::new(0);
static STRAY_CALLS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn record(value: i32, user: *mut c_void) -> i32 {
    if user.is_null() {
        NULL_USER_CALLS.fetch_add(1, Ordering::SeqCst);
        return 0;
    }
    let seen = unsafe { &mut *(user as *mut Seen) };
    seen.values.push(value);
    (value == seen.stop_at) as i32
}

extern "C" fn stray(_value: i32, _user: *mut c_void) -> i32 {
    STRAY_CALLS.fetch_add(1, Ordering::SeqCst);
    0
}

fn run(items: &[i32], stop_at: i32) -> (usize, Vec<i32>) {
    let mut seen = Seen { values: Vec::new(), stop_at };
    let cb: VisitFn = Some(record);
    let n = unsafe {
        for_each_until(items.as_ptr(), items.len(), cb, &mut seen as *mut Seen as *mut c_void)
    };
    (n, seen.values)
}

#[test]
fn exact_rust_signature() {
    let _: unsafe extern "C" fn(*const i32, usize, VisitFn, *mut c_void) -> usize = for_each_until;
    // VisitFn must be able to hold C's NULL.
    let none: VisitFn = None;
    assert!(none.is_none());
    assert_eq!(core::mem::size_of::<VisitFn>(), core::mem::size_of::<*const c_void>(),
               "VisitFn is pointer-sized, like C's visit_fn");
}

#[test]
fn visits_in_order_and_forwards_user() {
    NULL_USER_CALLS.store(0, Ordering::SeqCst);
    assert_eq!(run(&[5, 6, 7], 999), (3, vec![5, 6, 7]), "no early stop");
    assert_eq!(NULL_USER_CALLS.load(Ordering::SeqCst), 0, "callback received a null user pointer");
}

#[test]
fn stops_right_after_nonzero_and_counts_that_call() {
    assert_eq!(run(&[1, 2, 3, 4, 5], 3), (3, vec![1, 2, 3]), "stop at the third element");
    assert_eq!(run(&[9, 2, 3], 9), (1, vec![9]), "stop at the first element");
    assert_eq!(run(&[1, 2, 3], 3), (3, vec![1, 2, 3]), "stop at the last element");
}

#[test]
fn null_callback_null_items_and_empty() {
    STRAY_CALLS.store(0, Ordering::SeqCst);
    let items = [1, 2, 3];
    let mut seen = Seen { values: Vec::new(), stop_at: 0 };
    let user = &mut seen as *mut Seen as *mut c_void;
    unsafe {
        assert_eq!(for_each_until(items.as_ptr(), 3, None, user), 0, "NULL callback");
        let cb: VisitFn = Some(stray);
        assert_eq!(for_each_until(core::ptr::null(), 3, cb, user), 0, "NULL items");
        assert_eq!(for_each_until(items.as_ptr(), 0, cb, user), 0, "zero length");
    }
    assert_eq!(STRAY_CALLS.load(Ordering::SeqCst), 0, "callback must not be called");
}

#[test]
fn c_abi_symbol() {
    // Called exactly as C would: the callback is a nullable pointer-sized value.
    extern "C" {
        #[link_name = "for_each_until"]
        fn c_for_each_until(items: *const i32, len: usize, cb: *const c_void, user: *mut c_void) -> usize;
    }
    let items = [4, 4, 8];
    let mut seen = Seen { values: Vec::new(), stop_at: 8 };
    let f: extern "C" fn(i32, *mut c_void) -> i32 = record;
    let n = unsafe {
        c_for_each_until(items.as_ptr(), 3, f as *const c_void, &mut seen as *mut Seen as *mut c_void)
    };
    assert_eq!((n, seen.values), (3, vec![4, 4, 8]));
    let n = unsafe { c_for_each_until(items.as_ptr(), 3, core::ptr::null(), core::ptr::null_mut()) };
    assert_eq!(n, 0, "NULL callback through the C ABI");
}
