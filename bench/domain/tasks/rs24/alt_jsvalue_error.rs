use wasm_bindgen::prelude::*;

// Result<_, JsValue>, with the error built as a real Error by converting a JsError.
#[wasm_bindgen(js_name = "parseHexColor")]
pub fn parse_hex_color(input: String) -> Result<Box<[u8]>, JsValue> {
    let fail = |why: &str| -> JsValue { JsError::new(&format!("{why}: {input:?}")).into() };
    let Some(rest) = input.strip_prefix('#') else { return Err(fail("missing '#'")) };
    if !rest.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(fail("non-hex digit"));
    }
    let expanded: String = match rest.len() {
        6 => rest.to_string(),
        3 => rest.chars().flat_map(|c| [c, c]).collect(),
        _ => return Err(fail("expected #rgb or #rrggbb")),
    };
    let mut rgb = [0u8; 3];
    for (i, slot) in rgb.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&expanded[2 * i..2 * i + 2], 16).map_err(|_| fail("bad digit"))?;
    }
    Ok(Box::new(rgb))
}
