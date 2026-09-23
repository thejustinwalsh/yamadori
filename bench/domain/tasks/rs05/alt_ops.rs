use std::ops::{Add, Mul};

/// A length in metres; ABI-identical to `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Meters(pub f64);

impl Add for Meters {
    type Output = Meters;
    fn add(self, o: Meters) -> Meters {
        Meters(self.0 + o.0)
    }
}

impl Mul<f64> for Meters {
    type Output = Meters;
    fn mul(self, k: f64) -> Meters {
        Meters(self.0 * k)
    }
}

#[no_mangle]
pub extern "C" fn meters_add(a: Meters, b: Meters) -> Meters {
    a + b
}

#[no_mangle]
pub extern "C" fn meters_scale(m: Meters, k: f64) -> Meters {
    m * k
}
