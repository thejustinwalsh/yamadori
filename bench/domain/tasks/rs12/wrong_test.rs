// Last error kept in a process-global Mutex, not per thread: one thread sees another thread's failure.
use std::ffi::{c_char, CStr};
use std::num::IntErrorKind;
use std::sync::Mutex;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

fn set_error(msg: String) {
    *LAST_ERROR.lock().unwrap() = Some(msg);
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
        set_error("null argument".into());
        return -1;
    }
    let Ok(text) = CStr::from_ptr(s).to_str() else {
        set_error("invalid UTF-8".into());
        return -2;
    };
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 2 {
        set_error("expected exactly one comma".into());
        return -3;
    }
    let r = parse_coord(parts[0]).and_then(|x| parse_coord(parts[1]).map(|y| Point { x, y }));
    match r {
        Ok(p) => {
            *out = p;
            *LAST_ERROR.lock().unwrap() = None;
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
    let guard = LAST_ERROR.lock().unwrap();
    match guard.as_deref() {
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
    }
}
