// A successful parse never clears the previous error, so a stale message is still reported.
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

fn fail(code: i32, msg: String) -> i32 {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
    code
}

fn parse_coord(part: &str) -> Result<i32, (i32, String)> {
    let t = part.trim();
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err((-3, format!("not an integer: {t:?}")));
    }
    t.parse::<i32>().map_err(|e| match e.kind() {
        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => (-4, format!("out of range: {t}")),
        _ => (-3, format!("not an integer: {t:?}")),
    })
}

#[no_mangle]
pub unsafe extern "C" fn parse_point(s: *const c_char, out: *mut Point) -> i32 {
    if s.is_null() || out.is_null() {
        return fail(-1, "null argument".into());
    }
    let Ok(text) = CStr::from_ptr(s).to_str() else {
        return fail(-2, "invalid UTF-8".into());
    };
    let Some((a, b)) = text.split_once(',') else {
        return fail(-3, "missing comma".into());
    };
    if b.contains(',') {
        return fail(-3, "too many commas".into());
    }
    let x = match parse_coord(a) {
        Ok(v) => v,
        Err((c, m)) => return fail(c, m),
    };
    let y = match parse_coord(b) {
        Ok(v) => v,
        Err((c, m)) => return fail(c, m),
    };
    *out = Point { x, y };
    0
}

#[no_mangle]
pub unsafe extern "C" fn last_error_message(buf: *mut c_char, cap: usize) -> isize {
    LAST_ERROR.with(|e| match e.borrow().as_deref() {
        None => {
            if !buf.is_null() && cap >= 1 {
                *buf = 0;
            }
            0
        }
        Some(msg) => {
            let n = msg.len();
            if buf.is_null() || cap < n + 1 {
                return -((n + 1) as isize);
            }
            std::ptr::copy_nonoverlapping(msg.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
            n as isize
        }
    })
}
