use std::mem::size_of;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Stats {
    pub count: u32,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

#[no_mangle]
pub extern "C" fn stats_size() -> usize {
    size_of::<Stats>()
}

/// # Safety
/// `values` must point at `len` doubles; `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn stats_compute(values: *const f64, len: usize, out: *mut Stats) -> i32 {
    let Some(out) = out.as_mut() else { return -1 };
    if len == 0 {
        return 1;
    }
    if values.is_null() {
        return -1;
    }
    let xs = std::slice::from_raw_parts(values, len);
    let (first, rest) = xs.split_first().unwrap();
    let (lo, hi, total) = rest.iter().fold((*first, *first, *first), |(lo, hi, s), &x| {
        (if x < lo { x } else { lo }, if x > hi { x } else { hi }, s + x)
    });
    *out = Stats { count: xs.len() as u32, min: lo, max: hi, mean: total / xs.len() as f64 };
    0
}
