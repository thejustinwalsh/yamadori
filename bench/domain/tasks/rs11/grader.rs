use super::*;

extern "C" {
    #[link_name = "hex_encode"]
    fn c_hex(data: *const u8, len: usize, out: *mut u8, cap: usize) -> usize;
}

const CANARY: u8 = 0xCC;

/// Calls with a buffer of `cap` usable bytes followed by 8 canary bytes.
fn call(data: &[u8], cap: usize) -> (usize, Vec<u8>) {
    let mut buf = vec![CANARY; cap + 8];
    let n = unsafe { c_hex(data.as_ptr(), data.len(), buf.as_mut_ptr(), cap) };
    (n, buf)
}

#[test]
fn rust_signature() {
    let _: unsafe extern "C" fn(*const u8, usize, *mut std::ffi::c_char, usize) -> usize = hex_encode;
}

#[test]
fn exact_fit_writes_digits_and_one_nul() {
    let data = [0x00u8, 0x0f, 0xa5, 0xff, 0x10];
    let (n, buf) = call(&data, 11);
    assert_eq!(n, 11, "required size is 2*len+1");
    assert_eq!(&buf[..10], b"000fa5ff10", "lowercase hex digits");
    assert_eq!(buf[10], 0, "NUL terminator");
    assert!(buf[11..].iter().all(|&b| b == CANARY), "wrote past cap: {:?}", &buf[11..]);
}

#[test]
fn roomy_buffer_touches_nothing_after_the_nul() {
    let (n, buf) = call(&[0xde, 0xad], 64);
    assert_eq!(n, 5);
    assert_eq!(&buf[..5], b"dead\0");
    assert!(buf[5..].iter().all(|&b| b == CANARY), "bytes after the NUL were modified");
}

#[test]
fn too_small_writes_nothing_and_reports_the_size() {
    let data = [1u8, 2, 3];
    for cap in [0usize, 1, 5, 6] {
        let (n, buf) = call(&data, cap);
        assert_eq!(n, 7, "required size with cap {}", cap);
        assert!(buf.iter().all(|&b| b == CANARY), "cap {} < 7 but the buffer was written: {:?}", cap, buf);
    }
}

#[test]
fn size_query_with_null_out() {
    let data = [9u8; 100];
    assert_eq!(unsafe { c_hex(data.as_ptr(), 100, core::ptr::null_mut(), 0) }, 201);
    assert_eq!(unsafe { c_hex(data.as_ptr(), 100, core::ptr::null_mut(), 1000) }, 201);
}

#[test]
fn empty_and_null_data() {
    let mut buf = [CANARY; 4];
    let n = unsafe { c_hex(core::ptr::null(), 0, buf.as_mut_ptr(), 4) };
    assert_eq!(n, 1, "empty input needs 1 byte");
    assert_eq!(buf, [0, CANARY, CANARY, CANARY], "empty input writes just the NUL");
    let mut buf = [CANARY; 16];
    let n = unsafe { c_hex(core::ptr::null(), 3, buf.as_mut_ptr(), 16) };
    assert_eq!(n, 0, "null data with len > 0");
    assert!(buf.iter().all(|&b| b == CANARY), "null data: nothing written");
}
