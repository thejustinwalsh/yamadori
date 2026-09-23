pub const VAL_INT: u32 = 1;
pub const VAL_FLOAT: u32 = 2;
pub const VAL_BYTES: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
pub union Payload {
    pub i: i64,
    pub f: f64,
    pub bytes: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Value {
    pub tag: u32,
    pub payload: Payload,
}

#[no_mangle]
pub unsafe extern "C" fn value_to_f64(v: *const Value, out: *mut f64) -> i32 {
    if v.is_null() || out.is_null() {
        return -1;
    }
    let v = &*v;
    let x = match v.tag {
        VAL_INT => v.payload.i as f64,
        VAL_FLOAT => v.payload.f,
        VAL_BYTES => return -2,
        _ => return -3,
    };
    *out = x;
    0
}
