// Divides by the squared length instead of the length.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n2 = v.iter().map(|x| x * x).sum::<f32>();
    if n2 == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / n2).collect()
}
