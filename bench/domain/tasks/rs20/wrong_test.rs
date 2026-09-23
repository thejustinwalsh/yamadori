// No rename: JavaScript sees `xor_cipher`, and `xorCipher` does not exist.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn xor_cipher(data: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() {
        return data.to_vec();
    }
    data.iter()
        .enumerate()
        .map(|(i, b)| b ^ key[i % key.len()])
        .collect()
}
