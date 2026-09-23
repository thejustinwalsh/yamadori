// i64 instead of u64: the declaration still says bigint, but words with the top bit set come back negative.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn rotl64(x: i64, n: u32) -> i64 {
    x.rotate_left(n)
}

#[wasm_bindgen]
pub fn popcount64(x: i64) -> u32 {
    x.count_ones()
}
