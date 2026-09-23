use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn rotl64(x: u64, n: u32) -> u64 {
    let s = n & 63;
    if s == 0 {
        x
    } else {
        (x << s) | (x >> (64 - s))
    }
}

#[wasm_bindgen]
pub fn popcount64(mut x: u64) -> u8 {
    let mut c = 0u8;
    while x != 0 {
        x &= x - 1;
        c += 1;
    }
    c
}
