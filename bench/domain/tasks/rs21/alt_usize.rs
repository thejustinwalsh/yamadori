use wasm_bindgen::prelude::*;

/// Option<usize> also crosses as `number | undefined`.
#[wasm_bindgen]
pub fn find_byte(haystack: Vec<u8>, needle: u8, from: usize) -> Option<usize> {
    (from..haystack.len()).find(|&i| haystack[i] == needle)
}
