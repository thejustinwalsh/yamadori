use wasm_bindgen::prelude::*;

/// Owned inputs (copied in by the glue), mutated in place and returned.
#[wasm_bindgen(js_name = "xorCipher")]
pub fn xor_cipher(mut data: Vec<u8>, key: Vec<u8>) -> Vec<u8> {
    if !key.is_empty() {
        for (b, k) in data.iter_mut().zip(key.iter().cycle()) {
            *b ^= k;
        }
    }
    data
}
