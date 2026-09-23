// Leaves the discriminants implicit (0,1,2,3,4): matches raw correctly but writes the wrong integer.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[no_mangle]
pub unsafe extern "C" fn log_level_from_raw(raw: u32, out: *mut LogLevel) -> i32 {
    if out.is_null() {
        return -2;
    }
    let level = match raw {
        0 => LogLevel::Trace,
        10 => LogLevel::Debug,
        20 => LogLevel::Info,
        30 => LogLevel::Warn,
        40 => LogLevel::Error,
        _ => return -1,
    };
    *out = level;
    0
}
