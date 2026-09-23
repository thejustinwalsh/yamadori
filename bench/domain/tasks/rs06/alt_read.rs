/// Mirrors `struct Header` from the C header.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Header {
    pub kind: u8,
    pub id: u32,
    pub sub: u8,
}

/// Mirrors `struct Record`; the nested Header keeps its own C layout.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Record {
    pub header: Header,
    pub values: [u16; 3],
    pub checksum: u64,
}

impl Record {
    fn sum(&self) -> u64 {
        let Header { kind, id, sub } = self.header;
        let [a, b, c] = self.values;
        [kind as u64, id as u64, sub as u64, a as u64, b as u64, c as u64].iter().sum()
    }
}

#[no_mangle]
pub unsafe extern "C" fn record_sum(r: *const Record) -> u64 {
    if r.is_null() { 0 } else { r.read().sum() }
}
