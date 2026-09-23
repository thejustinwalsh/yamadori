use super::*;
use core::mem::{align_of, offset_of, size_of};

#[test]
fn header_layout_matches_c() {
    assert_eq!(offset_of!(Header, kind), 0, "Header.kind");
    assert_eq!(offset_of!(Header, id), 4, "Header.id");
    assert_eq!(offset_of!(Header, sub), 8, "Header.sub");
    assert_eq!(size_of::<Header>(), 12, "sizeof(struct Header)");
    assert_eq!(align_of::<Header>(), 4, "alignof(struct Header)");
}

#[test]
fn record_layout_matches_c() {
    assert_eq!(offset_of!(Record, header), 0, "Record.header");
    assert_eq!(offset_of!(Record, values), 12, "Record.values");
    assert_eq!(offset_of!(Record, checksum), 24, "Record.checksum");
    assert_eq!(size_of::<Record>(), 32, "sizeof(struct Record)");
    assert_eq!(align_of::<Record>(), 8, "alignof(struct Record)");
}

/// Bytes exactly as a C compiler lays out the struct, with junk in the padding.
fn c_bytes() -> [u64; 4] {
    let mut words = [0u64; 4];
    let b: &mut [u8; 32] = unsafe { &mut *(words.as_mut_ptr() as *mut [u8; 32]) };
    b.fill(0xA5); // padding bytes are garbage in C
    b[0] = 7; // kind
    b[4..8].copy_from_slice(&100_000u32.to_ne_bytes()); // id
    b[8] = 9; // sub
    b[12..14].copy_from_slice(&1000u16.to_ne_bytes());
    b[14..16].copy_from_slice(&2000u16.to_ne_bytes());
    b[16..18].copy_from_slice(&3000u16.to_ne_bytes());
    b[24..32].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_ne_bytes()); // checksum
    words
}

#[test]
fn record_sum_reads_c_memory() {
    extern "C" {
        fn record_sum(r: *const u8) -> u64;
    }
    let words = c_bytes();
    let got = unsafe { record_sum(words.as_ptr() as *const u8) };
    assert_eq!(got, 7 + 100_000 + 9 + 1000 + 2000 + 3000, "record_sum over C-laid-out bytes");
    assert_eq!(unsafe { record_sum(core::ptr::null()) }, 0, "null record");
}

#[test]
fn record_sum_rust_side() {
    let f: unsafe extern "C" fn(*const Record) -> u64 = record_sum;
    let r = Record {
        header: Header { kind: 255, id: u32::MAX, sub: 255 },
        values: [u16::MAX, 1, 2],
        checksum: 42,
    };
    let c = r; // Copy
    let _ = format!("{:?}", c);
    let want = 255u64 + u32::MAX as u64 + 255 + u16::MAX as u64 + 3;
    assert_eq!(unsafe { f(&r) }, want, "no narrow-integer overflow");
}
