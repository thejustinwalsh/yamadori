/// Linear-scan interner over owned Strings (no hashing).
pub struct Interner {
    items: Vec<String>,
}

#[no_mangle]
pub extern "C" fn interner_new() -> *mut Interner {
    Box::into_raw(Box::new(Interner { items: Vec::with_capacity(4) }))
}

#[no_mangle]
pub unsafe extern "C" fn interner_intern(h: *mut Interner, ptr: *const u8, len: usize) -> u32 {
    if h.is_null() || (ptr.is_null() && len > 0) {
        return u32::MAX;
    }
    let raw = if len == 0 { Vec::new() } else { std::slice::from_raw_parts(ptr, len).to_vec() };
    let s = match String::from_utf8(raw) {
        Ok(s) => s,
        Err(_) => return u32::MAX,
    };
    let h = &mut *h;
    if let Some(i) = h.items.iter().position(|x| *x == s) {
        return i as u32;
    }
    h.items.push(s);
    (h.items.len() - 1) as u32
}

#[no_mangle]
pub unsafe extern "C" fn interner_get(h: *const Interner, id: u32, out_len: *mut usize) -> *const u8 {
    if h.is_null() || out_len.is_null() {
        return std::ptr::null();
    }
    let h = &*h;
    if (id as usize) >= h.items.len() {
        return std::ptr::null();
    }
    let s = &h.items[id as usize];
    out_len.write(s.len());
    s.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn interner_len(h: *const Interner) -> u32 {
    if h.is_null() {
        0
    } else {
        let h = &*h;
        h.items.len() as u32
    }
}

#[no_mangle]
pub unsafe extern "C" fn interner_free(h: *mut Interner) {
    if !h.is_null() {
        let _ = Box::from_raw(h);
    }
}
