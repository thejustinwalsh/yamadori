// Reaches for js-sys, which is not a dependency of the crate.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn summarize(values: &[f64]) -> JsValue {
    if values.is_empty() {
        return JsValue::NULL;
    }
    let obj = js_sys::Object::new();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    js_sys::Reflect::set(&obj, &"count".into(), &(values.len() as f64).into()).unwrap();
    js_sys::Reflect::set(&obj, &"mean".into(), &mean.into()).unwrap();
    obj.into()
}
