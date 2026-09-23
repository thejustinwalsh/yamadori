// histogram_new never returns null: bins == 0 yields a live, empty histogram.
pub struct Histogram {
    counts: Vec<u64>,
}

#[no_mangle]
pub extern "C" fn histogram_new(bins: u32) -> *mut Histogram {
    Box::into_raw(Box::new(Histogram { counts: vec![0; bins as usize] }))
}

#[no_mangle]
pub unsafe extern "C" fn histogram_add(h: *mut Histogram, bin: u32) -> i32 {
    let Some(h) = h.as_mut() else { return -1 };
    match h.counts.get_mut(bin as usize) {
        Some(c) => {
            *c += 1;
            0
        }
        None => -2,
    }
}

#[no_mangle]
pub unsafe extern "C" fn histogram_count(h: *const Histogram, bin: u32) -> u64 {
    match h.as_ref() {
        Some(h) => h.counts.get(bin as usize).copied().unwrap_or(0),
        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn histogram_free(h: *mut Histogram) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}
