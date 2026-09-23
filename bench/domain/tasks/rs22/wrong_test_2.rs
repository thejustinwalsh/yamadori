// f64 "because JavaScript numbers are doubles": the API takes number, not bigint, and loses bits above 2^53.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn rotl64(x: f64, n: u32) -> f64 {
    (x as u64).rotate_left(n) as f64
}

#[wasm_bindgen]
pub fn popcount64(x: f64) -> u32 {
    (x as u64).count_ones()
}
