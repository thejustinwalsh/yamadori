use std::ptr::NonNull;

/// Opaque to C.
pub struct Histogram {
    bins: Box<[u64]>,
}

impl Histogram {
    fn bin(&mut self, i: u32) -> Option<&mut u64> {
        self.bins.get_mut(usize::try_from(i).ok()?)
    }
}

#[no_mangle]
pub extern "C" fn histogram_new(bins: u32) -> *mut Histogram {
    if bins == 0 {
        return std::ptr::null_mut();
    }
    let h = Box::new(Histogram { bins: vec![0u64; bins as usize].into_boxed_slice() });
    Box::leak(h) as *mut Histogram
}

#[no_mangle]
pub unsafe extern "C" fn histogram_add(h: *mut Histogram, bin: u32) -> i32 {
    let Some(mut h) = NonNull::new(h) else { return -1 };
    if let Some(slot) = h.as_mut().bin(bin) {
        *slot += 1;
        0
    } else {
        -2
    }
}

#[no_mangle]
pub unsafe extern "C" fn histogram_count(h: *const Histogram, bin: u32) -> u64 {
    if h.is_null() {
        return 0;
    }
    let h = &*h;
    if (bin as usize) < h.bins.len() { h.bins[bin as usize] } else { 0 }
}

#[no_mangle]
pub unsafe extern "C" fn histogram_free(h: *mut Histogram) {
    if let Some(p) = NonNull::new(h) {
        let _owned: Box<Histogram> = Box::from_raw(p.as_ptr());
    }
}
