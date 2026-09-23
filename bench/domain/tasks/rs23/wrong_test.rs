// `new` is not marked as the constructor: JavaScript gets a static Accumulator.new() and `new Accumulator(x)` is unusable.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Accumulator {
    total: f64,
    count: u32,
}

#[wasm_bindgen]
impl Accumulator {
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

    #[wasm_bindgen(getter)]
    pub fn total(&self) -> f64 {
        self.total
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> u32 {
        self.count
    }
}
