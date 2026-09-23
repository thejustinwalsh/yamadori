// Returns an exported class instance: its fields are prototype getters, so it is not a plain object.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Summary {
    pub count: u32,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

#[wasm_bindgen]
pub fn summarize(values: &[f64]) -> Option<Summary> {
    if values.is_empty() {
        return None;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Some(Summary { count: values.len() as u32, min, max, mean })
}
