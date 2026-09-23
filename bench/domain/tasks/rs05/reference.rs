#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meters(pub f64);

#[no_mangle]
pub extern "C" fn meters_add(a: Meters, b: Meters) -> Meters {
    Meters(a.0 + b.0)
}

#[no_mangle]
pub extern "C" fn meters_scale(m: Meters, k: f64) -> Meters {
    Meters(m.0 * k)
}
