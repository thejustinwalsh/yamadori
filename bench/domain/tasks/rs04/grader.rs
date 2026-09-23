use super::*;
use core::mem::{align_of, size_of};

#[test]
fn enum_is_a_u32_with_the_c_values() {
    assert_eq!(size_of::<LogLevel>(), 4, "log_level_t is 4 bytes");
    assert_eq!(align_of::<LogLevel>(), 4, "log_level_t is 4-aligned");
    let all = [
        (LogLevel::Trace, 0u32),
        (LogLevel::Debug, 10),
        (LogLevel::Info, 20),
        (LogLevel::Warn, 30),
        (LogLevel::Error, 40),
    ];
    for (l, v) in all {
        assert_eq!(l as u32, v, "{:?} as u32", l);
        let c = l; // Clone + Copy
        assert_eq!(c, l);
    }
}

#[test]
fn from_raw_through_the_c_abi() {
    // Declared as C sees it: out is a pointer to a 4-byte integer.
    extern "C" {
        fn log_level_from_raw(raw: u32, out: *mut u32) -> i32;
    }
    for v in [0u32, 10, 20, 30, 40] {
        let mut slot: u32 = 0xDEAD_BEEF;
        let rc = unsafe { log_level_from_raw(v, &mut slot) };
        assert_eq!(rc, 0, "raw {} is valid", v);
        assert_eq!(slot, v, "C reads back the 4-byte value for raw {}", v);
    }
    for v in [1u32, 5, 39, 41, 255, 256, u32::MAX] {
        let mut slot: u32 = 0xDEAD_BEEF;
        let rc = unsafe { log_level_from_raw(v, &mut slot) };
        assert_eq!(rc, -1, "raw {} is invalid", v);
        assert_eq!(slot, 0xDEAD_BEEF, "out untouched for invalid raw {}", v);
    }
    assert_eq!(unsafe { log_level_from_raw(30, core::ptr::null_mut()) }, -2, "null out");
}

#[test]
fn from_raw_as_rust_sees_it() {
    let f: unsafe extern "C" fn(u32, *mut LogLevel) -> i32 = log_level_from_raw;
    let mut l = LogLevel::Trace;
    assert_eq!(unsafe { f(40, &mut l) }, 0);
    assert_eq!(l, LogLevel::Error);
    assert_eq!(unsafe { f(7, &mut l) }, -1);
    assert_eq!(l, LogLevel::Error, "untouched on error");
}
