use std::cell::Cell;
use std::ffi::{c_char, CStr, CString};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

thread_local! {
    // Stored with its NUL already attached.
    static LAST: Cell<Option<CString>> = const { Cell::new(None) };
}

enum Fail {
    Syntax,
    Range,
}

/// Hand-rolled: accumulate in i64 and bail out as soon as the magnitude passes i32.
fn coord(raw: &str) -> Result<i32, Fail> {
    let t = raw.trim_matches(|c: char| c.is_ascii_whitespace()).as_bytes();
    let (neg, digits) = match t.first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() || digits.iter().any(|b| !b.is_ascii_digit()) {
        return Err(Fail::Syntax);
    }
    let limit: i64 = if neg { 1 << 31 } else { (1 << 31) - 1 };
    let mut v: i64 = 0;
    for d in digits {
        v = v * 10 + (d - b'0') as i64;
        if v > limit {
            return Err(Fail::Range);
        }
    }
    Ok(if neg { (-v) as i32 } else { v as i32 })
}

fn fail(code: i32, msg: &str) -> i32 {
    LAST.with(|l| l.set(Some(CString::new(msg).unwrap())));
    code
}

#[no_mangle]
pub unsafe extern "C" fn parse_point(s: *const c_char, out: *mut Point) -> i32 {
    if s.is_null() || out.is_null() {
        return fail(-1, "parse_point: null pointer");
    }
    let text = match CStr::from_ptr(s).to_str() {
        Ok(t) => t,
        Err(_) => return fail(-2, "parse_point: invalid UTF-8"),
    };
    let Some((a, b)) = text.split_once(',') else {
        return fail(-3, "parse_point: missing comma");
    };
    if b.contains(',') {
        return fail(-3, "parse_point: more than one comma");
    }
    let x = match coord(a) {
        Ok(v) => v,
        Err(Fail::Syntax) => return fail(-3, "parse_point: bad x"),
        Err(Fail::Range) => return fail(-4, "parse_point: x out of range"),
    };
    let y = match coord(b) {
        Ok(v) => v,
        Err(Fail::Syntax) => return fail(-3, "parse_point: bad y"),
        Err(Fail::Range) => return fail(-4, "parse_point: y out of range"),
    };
    out.write(Point { x, y });
    LAST.with(|l| l.set(None));
    0
}

#[no_mangle]
pub unsafe extern "C" fn last_error_message(buf: *mut c_char, cap: usize) -> isize {
    let cur = LAST.with(|l| l.take());
    let r = match &cur {
        None => {
            if !buf.is_null() && cap > 0 {
                buf.write(0);
            }
            0
        }
        Some(m) => {
            let bytes = m.as_bytes_with_nul();
            if buf.is_null() || cap < bytes.len() {
                -(bytes.len() as isize)
            } else {
                std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, buf, bytes.len());
                (bytes.len() - 1) as isize
            }
        }
    };
    LAST.with(|l| l.set(cur));
    r
}
