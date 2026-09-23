use std::collections::HashMap;

pub struct Interner {
    // Each string owns its own heap buffer, so growing `strings` moves the
    // Box<str> handles but never the bytes they point to.
    strings: Vec<Box<str>>,
    ids: HashMap<Box<str>, u32>,
}

#[no_mangle]
pub extern "C" fn interner_new() -> *mut Interner {
    Box::into_raw(Box::new(Interner { strings: Vec::new(), ids: HashMap::new() }))
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
    if let Some(&id) = h.ids.get(s) {
        return id;
    }
    let id = h.strings.len() as u32;
    h.strings.push(Box::from(s));
    h.ids.insert(Box::from(s), id);
    id
}

#[no_mangle]
pub unsafe extern "C" fn interner_get(h: *const Interner, id: u32, out_len: *mut usize) -> *const u8 {
    let Some(h) = h.as_ref() else { return std::ptr::null() };
    if out_len.is_null() {
        return std::ptr::null();
    }
    match h.strings.get(id as usize) {
        Some(s) => {
            *out_len = s.len();
            s.as_ptr()
        }
        None => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn interner_len(h: *const Interner) -> u32 {
    h.as_ref().map_or(0, |h| h.strings.len() as u32)
}

#[no_mangle]
pub unsafe extern "C" fn interner_free(h: *mut Interner) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}
