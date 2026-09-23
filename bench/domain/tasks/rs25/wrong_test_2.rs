// Empty input returns JsValue::UNDEFINED instead of null.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Object)]
    fn new_object() -> JsValue;

    #[wasm_bindgen(js_namespace = Reflect, js_name = set)]
    fn reflect_set(target: &JsValue, key: &JsValue, value: &JsValue) -> bool;
}

#[wasm_bindgen]
pub fn summarize(values: &[f64]) -> JsValue {
    if values.is_empty() {
        return JsValue::UNDEFINED;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let obj = new_object();
    for (k, v) in [("count", values.len() as f64), ("min", min), ("max", max), ("mean", mean)] {
        reflect_set(&obj, &k.into(), &v.into());
    }
    obj
}
