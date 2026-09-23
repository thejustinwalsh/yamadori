use core::convert::TryFrom;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LogLevel {
    Trace = 0,
    Debug = 10,
    Info = 20,
    Warn = 30,
    Error = 40,
}

impl TryFrom<u32> for LogLevel {
    type Error = ();
    fn try_from(v: u32) -> Result<Self, ()> {
        const ALL: [LogLevel; 5] = [LogLevel::Trace, LogLevel::Debug, LogLevel::Info,
                                    LogLevel::Warn, LogLevel::Error];
        ALL.iter().copied().find(|l| *l as u32 == v).ok_or(())
    }
}

/// # Safety
/// `out` must be null or valid for a write of one `LogLevel`.
#[no_mangle]
pub unsafe extern "C" fn log_level_from_raw(raw: u32, out: *mut LogLevel) -> i32 {
    let Some(slot) = out.as_mut() else { return -2 };
    match LogLevel::try_from(raw) {
        Ok(l) => {
            *slot = l;
            0
        }
        Err(()) => -1,
    }
}
