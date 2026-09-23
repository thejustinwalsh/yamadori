use super::*;
use core::mem::{offset_of, size_of};

extern "C" {
    #[link_name = "parse_point"]
    fn c_parse(s: *const u8, out: *mut [i32; 2]) -> i32;
    #[link_name = "last_error_message"]
    fn c_last(buf: *mut u8, cap: usize) -> isize;
}

const UNTOUCHED: [i32; 2] = [0x5A5A_5A5A, -0x5A5A_5A5A];

fn parse(s: &[u8]) -> (i32, [i32; 2]) {
    let mut z = s.to_vec();
    z.push(0);
    let mut out = UNTOUCHED;
    let rc = unsafe { c_parse(z.as_ptr(), &mut out) };
    (rc, out)
}

fn clear() {
    assert_eq!(parse(b"0,0").0, 0, "parse_point(\"0,0\") must succeed");
}

/// The current message, read with an exactly sized buffer.
fn message() -> Option<String> {
    let need = unsafe { c_last(core::ptr::null_mut(), 0) };
    if need == 0 {
        return None;
    }
    assert!(need < -1, "with an error pending and no buffer, expected -(N+1) with N >= 1, got {}", need);
    let size = (-need) as usize;
    let mut buf = vec![0xCCu8; size + 4];
    let n = unsafe { c_last(buf.as_mut_ptr(), size) };
    assert_eq!(n, need.abs() - 1, "return value is N when the buffer fits");
    assert_eq!(buf[size - 1], 0, "NUL terminator at [N]");
    assert!(buf[size..].iter().all(|&b| b == 0xCC), "wrote past cap");
    let msg = &buf[..size - 1];
    assert!(!msg.contains(&0), "message has an interior NUL");
    Some(String::from_utf8_lossy(msg).into_owned())
}

#[test]
fn point_layout_and_signatures() {
    assert_eq!(size_of::<Point>(), 8);
    assert_eq!(offset_of!(Point, x), 0);
    assert_eq!(offset_of!(Point, y), 4);
    let _: unsafe extern "C" fn(*const std::ffi::c_char, *mut Point) -> i32 = parse_point;
    let _: unsafe extern "C" fn(*mut std::ffi::c_char, usize) -> isize = last_error_message;
    let mut p = Point { x: 1, y: 2 };
    let rc = unsafe { parse_point(b"-3,+4\0".as_ptr() as *const std::ffi::c_char, &mut p) };
    assert_eq!((rc, p), (0, Point { x: -3, y: 4 }));
}

#[test]
fn parses_valid_input() {
    for (s, want) in [
        (&b"3,4"[..], [3, 4]),
        (b" 12 ,\t-7 ", [12, -7]),
        (b"+0,-0", [0, 0]),
        (b"2147483647,-2147483648", [i32::MAX, i32::MIN]),
        (b"007,08", [7, 8]),
    ] {
        assert_eq!(parse(s), (0, want), "input {:?}", String::from_utf8_lossy(s));
    }
}

#[test]
fn error_codes_and_out_untouched() {
    for (s, code) in [
        (&b""[..], -3),
        (b"3", -3),
        (b"3,", -3),
        (b",4", -3),
        (b"1,2,3", -3),
        (b"a,1", -3),
        (b"1.5,2", -3),
        (b"+,1", -3),
        (b"1 2,3", -3),
        (b"--1,2", -3),
        (b"2147483648,0", -4),
        (b"0,-2147483649", -4),
        (b"99999999999999999999999999999999999999999,1", -4),
        (b"1,-99999999999999999999999999999999999999999", -4),
        (b"\xFF,1", -2),
    ] {
        assert_eq!(parse(s), (code, UNTOUCHED), "input {:?}", String::from_utf8_lossy(s));
    }
    let mut out = UNTOUCHED;
    unsafe {
        assert_eq!(c_parse(core::ptr::null(), &mut out), -1, "null s");
        assert_eq!(c_parse(b"1,2\0".as_ptr(), core::ptr::null_mut()), -1, "null out");
    }
    assert_eq!(out, UNTOUCHED);
}

#[test]
fn message_protocol() {
    clear();
    assert_eq!(message(), None, "no error after a success");
    let mut one = [0xCCu8; 2];
    assert_eq!(unsafe { c_last(one.as_mut_ptr(), 1) }, 0);
    assert_eq!(one, [0, 0xCC], "no error: writes a single NUL");

    assert_eq!(parse(b"x,y").0, -3);
    let m = message().expect("an error message after a failed parse");
    assert!(!m.is_empty());
    let need = unsafe { c_last(core::ptr::null_mut(), 0) };
    let mut small = vec![0xCCu8; 8 + m.len()];
    let rc = unsafe { c_last(small.as_mut_ptr(), m.len()) }; // one byte short
    assert_eq!(rc, need, "too small by one: returns -(N+1)");
    assert!(small.iter().all(|&b| b == 0xCC), "too small: nothing written");

    clear();
    assert_eq!(message(), None, "a successful parse clears the error");
}

#[test]
fn errors_are_per_thread() {
    clear();
    assert_eq!(parse(b"nope").0, -3);
    assert!(message().is_some());
    let other = std::thread::spawn(|| message()).join().unwrap();
    assert_eq!(other, None, "a fresh thread must not see this thread's error");

    clear();
    assert_eq!(message(), None, "a successful parse clears the error");
    std::thread::spawn(|| {
        assert_eq!(parse(b"1,99999999999").0, -4);
    })
    .join()
    .unwrap();
    assert_eq!(message(), None, "another thread's failure must not show up here");
}
