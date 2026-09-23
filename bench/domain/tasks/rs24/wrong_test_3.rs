// Short form parsed as single nibbles: #abc becomes [0x0a, 0x0b, 0x0c] instead of [0xaa, 0xbb, 0xcc].
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = parseHexColor)]
pub fn parse_hex_color(s: &str) -> Result<Vec<u8>, JsError> {
    let bad = || JsError::new(&format!("invalid hex color: {s}"));
    let digits = s.strip_prefix('#').ok_or_else(bad)?;
    if !digits.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(bad());
    }
    match digits.len() {
        3 => digits
            .chars()
            .map(|c| c.to_digit(16).map(|d| d as u8).ok_or_else(bad))
            .collect(),
        6 => (0..3)
            .map(|i| u8::from_str_radix(&digits[2 * i..2 * i + 2], 16).map_err(|_| bad()))
            .collect(),
        _ => Err(bad()),
    }
}
