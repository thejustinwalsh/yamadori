use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = JSON, js_name = parse)]
    fn json_parse(text: &str) -> JsValue;
}

/// Builds the object as JSON text and lets the JS engine parse it.
#[wasm_bindgen]
pub fn summarize(values: Vec<f64>) -> JsValue {
    if values.is_empty() {
        return json_parse("null");
    }
    let (mut lo, mut hi, mut sum) = (values[0], values[0], 0.0);
    for &v in &values {
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v;
    }
    // `{:?}` prints f64 with enough digits to round-trip exactly.
    json_parse(&format!(
        "{{\"count\":{},\"min\":{:?},\"max\":{:?},\"mean\":{:?}}}",
        values.len(),
        lo,
        hi,
        sum / values.len() as f64
    ))
}
