// repr(C, packed): no padding at all, which is not what the C compiler emits for this header.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub kind: u8,
    pub id: u32,
    pub sub: u8,
}

#[repr(C, packed)]
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
    let r = core::ptr::read_unaligned(r);
    let h = r.header;
    let vals = r.values;
    h.kind as u64 + h.id as u64 + h.sub as u64 + vals.iter().map(|&v| v as u64).sum::<u64>()
}
