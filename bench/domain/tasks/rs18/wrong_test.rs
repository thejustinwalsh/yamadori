// Stores every string in one growing String arena: when it reallocates, previously returned pointers move.
pub struct Interner {
    arena: String,
    spans: Vec<(usize, usize)>,
}

#[no_mangle]
pub extern "C" fn interner_new() -> *mut Interner {
    Box::into_raw(Box::new(Interner { arena: String::new(), spans: Vec::new() }))
}

#[no_mangle]
pub unsafe extern "C" fn interner_intern(h: *mut Interner, ptr: *const u8, len: usize) -> u32 {
    let Some(h) = h.as_mut() else { return u32::MAX };
    let bytes: &[u8] = if len == 0 {
        &[]
    } else if ptr.is_null() {
        return u32::MAX;
    } else {
        std::slice::from_raw_parts(ptr, len)
    };
    let Ok(s) = std::str::from_utf8(bytes) else { return u32::MAX };
    for (i, &(start, n)) in h.spans.iter().enumerate() {
        if &h.arena[start..start + n] == s {
            return i as u32;
        }
    }
    let start = h.arena.len();
    h.arena.push_str(s);
    h.spans.push((start, s.len()));
    (h.spans.len() - 1) as u32
}

#[no_mangle]
pub unsafe extern "C" fn interner_get(h: *const Interner, id: u32, out_len: *mut usize) -> *const u8 {
    let Some(h) = h.as_ref() else { return std::ptr::null() };
    if out_len.is_null() {
        return std::ptr::null();
    }
    match h.spans.get(id as usize) {
        Some(&(start, n)) => {
            *out_len = n;
            h.arena.as_ptr().add(start)
        }
        None => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn interner_len(h: *const Interner) -> u32 {
    h.as_ref().map_or(0, |h| h.spans.len() as u32)
}

#[no_mangle]
pub unsafe extern "C" fn interner_free(h: *mut Interner) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}
