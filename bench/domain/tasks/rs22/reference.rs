use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn rotl64(x: u64, n: u32) -> u64 {
    x.rotate_left(n)
}

#[wasm_bindgen]
pub fn popcount64(x: u64) -> u32 {
    x.count_ones()
}
