use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn find_byte(haystack: &[u8], needle: u8, from: u32) -> Option<u32> {
    let start = from as usize;
    if start >= haystack.len() {
        return None;
    }
    haystack[start..]
        .iter()
        .position(|&b| b == needle)
        .map(|i| (start + i) as u32)
}
