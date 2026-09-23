use core::ptr;

pub const VAL_INT: u32 = 1;
pub const VAL_FLOAT: u32 = 2;
pub const VAL_BYTES: u32 = 3;

#[derive(Clone, Copy)]
#[repr(C)]
pub union Payload {
    pub i: i64,
    pub f: f64,
    pub bytes: [u8; 8],
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Value {
    pub tag: u32,
    pub payload: Payload,
}

impl Value {
    fn as_f64(&self) -> Result<f64, i32> {
        // SAFETY: the tag says which field is live; every field is plain old data.
        unsafe {
            if self.tag == VAL_INT {
                Ok(self.payload.i as f64)
            } else if self.tag == VAL_FLOAT {
                Ok(self.payload.f)
            } else if self.tag == VAL_BYTES {
                Err(-2)
            } else {
                Err(-3)
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn value_to_f64(v: *const Value, out: *mut f64) -> i32 {
    match (v.as_ref(), out.is_null()) {
        (Some(v), false) => match v.as_f64() {
            Ok(x) => {
                ptr::write(out, x);
                0
            }
            Err(code) => code,
        },
        _ => -1,
    }
}
