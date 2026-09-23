// repr(C) on the outer struct only: Header is repr(Rust), so rustc packs it to 8 bytes (id, kind, sub).
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub kind: u8,
    pub id: u32,
    pub sub: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Record {
    pub header: Header,
    pub values: [u16; 3],
    pub checksum: u64,
}

#[no_mangle]
pub unsafe extern "C" fn record_sum(r: *const Record) -> u64 {
    if r.is_null() {
        return 0;
    }
    let r = &*r;
    let mut s = r.header.kind as u64 + r.header.id as u64 + r.header.sub as u64;
    for v in r.values {
        s += v as u64;
    }
    s
}
