// Right maths, but takes Vec<f64>: the JS type becomes Float64Array.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn normalize(v: Vec<f64>) -> Vec<f64> {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n == 0.0 {
        return v;
    }
    v.iter().map(|x| x / n).collect()
}
