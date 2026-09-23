use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum().sqrt();
    v.iter().map(|x| x / n).collect()
}
