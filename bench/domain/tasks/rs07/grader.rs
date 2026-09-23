use super::*;
use core::mem::{align_of, offset_of, size_of};

#[test]
fn layout_matches_c() {
    assert_eq!(size_of::<Payload>(), 8, "sizeof(union Payload)");
    assert_eq!(align_of::<Payload>(), 8, "alignof(union Payload)");
    assert_eq!(offset_of!(Value, tag), 0, "Value.tag");
    assert_eq!(offset_of!(Value, payload), 8, "Value.payload");
    assert_eq!(size_of::<Value>(), 16, "sizeof(struct Value)");
    assert_eq!(align_of::<Value>(), 8, "alignof(struct Value)");
    assert_eq!((VAL_INT, VAL_FLOAT, VAL_BYTES), (1, 2, 3));
    let p = Payload { bytes: [1, 2, 3, 4, 5, 6, 7, 8] };
    let q = p; // Copy
    assert_eq!(unsafe { q.i }.to_ne_bytes(), [1, 2, 3, 4, 5, 6, 7, 8], "fields overlap at offset 0");
}

/// A Value exactly as C writes it: u32 tag, 4 bytes of garbage padding, payload.
fn c_value(tag: u32, payload: [u8; 8]) -> [u64; 2] {
    let mut words = [0u64; 2];
    let b: &mut [u8; 16] = unsafe { &mut *(words.as_mut_ptr() as *mut [u8; 16]) };
    b[0..4].copy_from_slice(&tag.to_ne_bytes());
    b[4..8].copy_from_slice(&[0xEE; 4]);
    b[8..16].copy_from_slice(&payload);
    words
}

extern "C" {
    #[link_name = "value_to_f64"]
    fn c_value_to_f64(v: *const u8, out: *mut f64) -> i32;
}

fn call(tag: u32, payload: [u8; 8]) -> (i32, f64) {
    let w = c_value(tag, payload);
    let mut out = -12345.0;
    let rc = unsafe { c_value_to_f64(w.as_ptr() as *const u8, &mut out) };
    (rc, out)
}

#[test]
fn converts_from_c_memory() {
    assert_eq!(call(1, 3i64.to_ne_bytes()), (0, 3.0), "VAL_INT 3");
    assert_eq!(call(1, (-7i64).to_ne_bytes()), (0, -7.0), "VAL_INT -7");
    assert_eq!(call(1, (1i64 << 40).to_ne_bytes()), (0, (1u64 << 40) as f64), "VAL_INT 2^40");
    assert_eq!(call(2, 2.5f64.to_ne_bytes()), (0, 2.5), "VAL_FLOAT 2.5");
    assert_eq!(call(3, [1; 8]), (-2, -12345.0), "VAL_BYTES leaves out untouched");
    assert_eq!(call(0, [0; 8]), (-3, -12345.0), "unknown tag 0");
    assert_eq!(call(99, [0; 8]), (-3, -12345.0), "unknown tag 99");
}

#[test]
fn null_arguments() {
    let w = c_value(2, 1.0f64.to_ne_bytes());
    let mut out = 5.0;
    unsafe {
        assert_eq!(c_value_to_f64(core::ptr::null(), &mut out), -1, "null v");
        assert_eq!(c_value_to_f64(w.as_ptr() as *const u8, core::ptr::null_mut()), -1, "null out");
    }
    assert_eq!(out, 5.0);
}

#[test]
fn rust_side_values() {
    let f: unsafe extern "C" fn(*const Value, *mut f64) -> i32 = value_to_f64;
    let v = Value { tag: VAL_INT, payload: Payload { i: 1 << 53 } };
    let mut out = 0.0;
    assert_eq!(unsafe { f(&v, &mut out) }, 0);
    assert_eq!(out, 9007199254740992.0);
    let v = Value { tag: VAL_FLOAT, payload: Payload { f: -0.25 } };
    assert_eq!(unsafe { f(&v, &mut out) }, 0);
    assert_eq!(out, -0.25);
}
