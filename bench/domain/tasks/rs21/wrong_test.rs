// C habit: returns -1 for "not found", so JavaScript gets a number, never undefined.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn find_byte(haystack: &[u8], needle: u8, from: u32) -> i32 {
    let start = from as usize;
    if start >= haystack.len() {
        return -1;
    }
    match haystack[start..].iter().position(|&b| b == needle) {
        Some(i) => (start + i) as i32,
        None => -1,
    }
}
