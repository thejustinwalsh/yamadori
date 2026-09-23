use wasm_bindgen::prelude::*;

fn hex(c: u8) -> Option<u8> {
    (c as char).to_digit(16).map(|d| d as u8)
}

#[wasm_bindgen(js_name = parseHexColor)]
pub fn parse_hex_color(s: &str) -> Result<Vec<u8>, JsError> {
    let bad = || JsError::new(&format!("invalid hex color: {s}"));
    let digits = s.strip_prefix('#').ok_or_else(bad)?.as_bytes();
    let nib: Vec<u8> = digits.iter().map(|&c| hex(c)).collect::<Option<_>>().ok_or_else(bad)?;
    match nib.len() {
        3 => Ok(nib.iter().map(|&d| d * 17).collect()),
        6 => Ok(nib.chunks(2).map(|p| p[0] * 16 + p[1]).collect()),
        _ => Err(bad()),
    }
}
