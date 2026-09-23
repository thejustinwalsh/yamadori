use super::*;
use core::mem::{align_of, offset_of, size_of};

#[test]
fn layout_matches_c() {
    assert_eq!(offset_of!(PacketHeader, tag), 0);
    assert_eq!(offset_of!(PacketHeader, len), 4);
    assert_eq!(offset_of!(PacketHeader, flags), 8);
    assert_eq!(size_of::<PacketHeader>(), 12);
    assert_eq!(align_of::<PacketHeader>(), 4);
}

#[test]
fn size_export_is_c_abi_and_unmangled() {
    extern "C" {
        fn packet_header_size() -> usize;
    }
    assert_eq!(unsafe { packet_header_size() }, 12);
    let h = PacketHeader { tag: 1, len: 2, flags: 3 };
    assert_eq!((h.tag, h.len, h.flags), (1, 2, 3));
}
