// Plain `pub` fields: JavaScript can assign acc.total = ..., and the declaration says `total: number;` (not readonly).
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Accumulator {
    pub total: f64,
    pub count: u32,
}

#[wasm_bindgen]
impl Accumulator {
    #[wasm_bindgen(constructor)]
    pub fn new(initial: f64) -> Accumulator {
        Accumulator { total: initial, count: 0 }
    }

    pub fn add(&mut self, x: f64) {
        self.total += x;
        self.count += 1;
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
        self.count = 0;
    }
}
