use wasm_bindgen::prelude::*;

/// Public fields exported as read-only properties instead of getter methods.
#[wasm_bindgen]
pub struct Accumulator {
    #[wasm_bindgen(readonly)]
    pub total: f64,
    #[wasm_bindgen(readonly)]
    pub count: u32,
}

#[wasm_bindgen]
impl Accumulator {
    #[wasm_bindgen(constructor)]
    pub fn create(initial: f64) -> Self {
        Self { total: initial, count: 0 }
    }

    pub fn add(&mut self, x: f64) {
        self.total += x;
        self.count += 1;
    }

    pub fn reset(&mut self) {
        *self = Self { total: 0.0, count: 0 };
    }
}
