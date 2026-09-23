use super::*;
use core::mem::{align_of, size_of};

// Fails to compile (grader stage `test`) if Meters has no guaranteed layout.
#[deny(improper_ctypes_definitions)]
extern "C" fn _ffi_safety_probe(m: Meters) -> Meters {
    m
}

#[test]
fn newtype_shape() {
    assert_eq!(size_of::<Meters>(), size_of::<f64>());
    assert_eq!(align_of::<Meters>(), align_of::<f64>());
    let m = Meters(2.5);
    let c = m;
    assert_eq!(c.0, 2.5);
    assert_eq!(m, c);
    assert_eq!(_ffi_safety_probe(Meters(1.0)), Meters(1.0));
}

#[test]
fn rust_signatures() {
    let add: extern "C" fn(Meters, Meters) -> Meters = meters_add;
    let scale: extern "C" fn(Meters, f64) -> Meters = meters_scale;
    assert_eq!(add(Meters(1.5), Meters(2.25)), Meters(3.75));
    assert_eq!(scale(Meters(3.0), -0.5), Meters(-1.5));
}

#[test]
fn callable_as_c_declares_it_with_plain_doubles() {
    extern "C" {
        fn meters_add(a: f64, b: f64) -> f64;
        fn meters_scale(m: f64, k: f64) -> f64;
    }
    unsafe {
        assert_eq!(meters_add(1.5, 2.25), 3.75, "meters_add(double, double)");
        assert_eq!(meters_add(-1e300, 1e300), 0.0, "meters_add(double, double)");
        assert_eq!(meters_scale(3.0, -0.5), -1.5, "meters_scale(double, double)");
        assert_eq!(meters_scale(0.125, 8.0), 1.0, "meters_scale(double, double)");
    }
}
