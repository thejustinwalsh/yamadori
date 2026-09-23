// total/count are plain methods, so JavaScript sees acc.total as a function, not a number.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Accumulator {
    total: f64,
    count: u32,
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

    pub fn total(&self) -> f64 {
        self.total
    }

    pub fn count(&self) -> u32 {
        self.count
    }
}
