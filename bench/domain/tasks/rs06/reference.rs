#[repr(C)]
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
    let Some(r) = r.as_ref() else { return 0 };
    let h = &r.header;
    h.kind as u64
        + h.id as u64
        + h.sub as u64
        + r.values.iter().map(|&v| v as u64).sum::<u64>()
}
