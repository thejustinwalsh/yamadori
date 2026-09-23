// Missing #[repr(C)]: rustc is free to reorder the fields, and does.
#[derive(Debug, Clone, Copy)]
pub struct PacketHeader {
    pub tag: u8,
    pub len: u32,
    pub flags: u16,
}

#[no_mangle]
pub extern "C" fn packet_header_size() -> usize {
    core::mem::size_of::<PacketHeader>()
}
