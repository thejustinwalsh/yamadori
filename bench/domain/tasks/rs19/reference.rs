#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub count: u32,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

#[no_mangle]
pub extern "C" fn stats_size() -> usize {
    core::mem::size_of::<Stats>()
}

#[no_mangle]
pub unsafe extern "C" fn stats_compute(values: *const f64, len: usize, out: *mut Stats) -> i32 {
    if out.is_null() || (values.is_null() && len > 0) {
        return -1;
    }
    if len == 0 {
        return 1;
    }
    let xs = core::slice::from_raw_parts(values, len);
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &x in xs {
        min = min.min(x);
        max = max.max(x);
        sum += x;
    }
    *out = Stats { count: len as u32, min, max, mean: sum / len as f64 };
    0
}
