use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::num::IntErrorKind;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
}

fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

fn parse_coord(part: &str) -> Result<i32, (i32, String)> {
    let t = part.trim_matches(|c: char| c.is_ascii_whitespace());
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err((-3, format!("not an integer: {t:?}")));
    }
    t.parse::<i32>().map_err(|e| match e.kind() {
        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
            (-4, format!("out of i32 range: {t}"))
        }
        _ => (-3, format!("not an integer: {t:?}")),
    })
}

fn parse(s: &str) -> Result<Point, (i32, String)> {
    let mut parts = s.split(',');
    let (Some(a), Some(b), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err((-3, "expected exactly one comma".to_string()));
    };
    let x = parse_coord(a)?;
    let y = parse_coord(b)?;
    Ok(Point { x, y })
}

#[no_mangle]
pub unsafe extern "C" fn parse_point(s: *const c_char, out: *mut Point) -> i32 {
    if s.is_null() || out.is_null() {
        set_error("null argument".to_string());
        return -1;
    }
    let Ok(text) = CStr::from_ptr(s).to_str() else {
        set_error("input is not valid UTF-8".to_string());
        return -2;
    };
    match parse(text) {
        Ok(p) => {
            *out = p;
            clear_error();
            0
        }
        Err((code, msg)) => {
            set_error(msg);
            code
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn last_error_message(buf: *mut c_char, cap: usize) -> isize {
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        let Some(msg) = e.as_deref() else {
            if !buf.is_null() && cap >= 1 {
                *buf = 0;
            }
            return 0;
        };
        let n = msg.len();
        if buf.is_null() || cap < n + 1 {
            return -((n + 1) as isize);
        }
        std::ptr::copy_nonoverlapping(msg.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
        n as isize
    })
}
