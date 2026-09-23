use wasm_bindgen::prelude::*;

// No js-sys: import the two JS globals needed to build an object by hand.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Object)]
    fn new_object() -> JsValue;

    #[wasm_bindgen(js_namespace = Reflect, js_name = set)]
    fn reflect_set(target: &JsValue, key: &JsValue, value: &JsValue) -> bool;
}

fn set(obj: &JsValue, key: &str, value: f64) {
    reflect_set(obj, &JsValue::from_str(key), &JsValue::from_f64(value));
}

#[wasm_bindgen]
pub fn summarize(values: &[f64]) -> JsValue {
    if values.is_empty() {
        return JsValue::NULL;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let obj = new_object();
    set(&obj, "count", values.len() as f64);
    set(&obj, "min", min);
    set(&obj, "max", max);
    set(&obj, "mean", mean);
    obj
}
