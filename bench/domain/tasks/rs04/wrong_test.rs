// #[repr(u8)] instead of u32: the enum is 1 byte, so C's 4-byte log_level_t reads 3 stale bytes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace = 0,
    Debug = 10,
    Info = 20,
    Warn = 30,
    Error = 40,
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
