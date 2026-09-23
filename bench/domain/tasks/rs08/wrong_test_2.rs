// histogram_free drops the raw pointer, not a Box: nothing is ever released.
pub struct Histogram {
    counts: Vec<u64>,
}

#[no_mangle]
pub extern "C" fn histogram_new(bins: u32) -> *mut Histogram {
    if bins == 0 {
        return core::ptr::null_mut();
    }
    Box::into_raw(Box::new(Histogram { counts: vec![0; bins as usize] }))
}

#[no_mangle]
pub unsafe extern "C" fn histogram_add(h: *mut Histogram, bin: u32) -> i32 {
    if h.is_null() {
        return -1;
    }
    let h = &mut *h;
    if bin as usize >= h.counts.len() {
        return -2;
    }
    h.counts[bin as usize] += 1;
    0
}

#[no_mangle]
pub unsafe extern "C" fn histogram_count(h: *const Histogram, bin: u32) -> u64 {
    if h.is_null() {
        return 0;
    }
    let h = &*h;
    h.counts.get(bin as usize).copied().unwrap_or(0)
}

#[no_mangle]
pub unsafe extern "C" fn histogram_free(h: *mut Histogram) {
    let _ = h;
}
