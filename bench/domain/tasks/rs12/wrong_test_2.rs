// Range check by parsing into i64: a number too big even for i64 is reported as a syntax error (-3), not -4.
use std::cell::RefCell;
use std::ffi::{c_char, CStr};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = RefCell::new(None);
}

fn parse_coord(part: &str) -> Result<i32, (i32, String)> {
    let t = part.trim();
    let wide: i64 = t.parse().map_err(|_| (-3, format!("not an integer: {t:?}")))?;
    i32::try_from(wide).map_err(|_| (-4, format!("out of range: {t}")))
}

#[no_mangle]
pub unsafe extern "C" fn parse_point(s: *const c_char, out: *mut Point) -> i32 {
    let r: Result<Point, (i32, String)> = (|| {
        if s.is_null() || out.is_null() {
            return Err((-1, "null argument".to_string()));
        }
        let text = CStr::from_ptr(s).to_str().map_err(|_| (-2, "invalid UTF-8".to_string()))?;
        let (a, b) = text.split_once(',').ok_or((-3, "missing comma".to_string()))?;
        if b.contains(',') {
            return Err((-3, "too many commas".to_string()));
        }
        Ok(Point { x: parse_coord(a)?, y: parse_coord(b)? })
    })();
    match r {
        Ok(p) => {
            *out = p;
            LAST_ERROR.with(|e| *e.borrow_mut() = None);
            0
        }
        Err((code, msg)) => {
            LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
            code
        }
    }
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
