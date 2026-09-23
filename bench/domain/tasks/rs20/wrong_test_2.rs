// zip() without cycle(): output stops at the key's length instead of repeating the key.
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = xorCipher)]
pub fn xor_cipher(data: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() {
        return data.to_vec();
    }
    data.iter().zip(key.iter()).map(|(b, k)| b ^ k).collect()
}
