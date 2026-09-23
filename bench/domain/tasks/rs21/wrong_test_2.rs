// Returns JsValue::NULL for "not found": JavaScript gets null (and the declaration says `any`).
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn find_byte(haystack: &[u8], needle: u8, from: u32) -> JsValue {
    let start = from as usize;
    if start < haystack.len() {
        if let Some(i) = haystack[start..].iter().position(|&b| b == needle) {
            return JsValue::from((start + i) as u32);
        }
    }
    JsValue::NULL
}
