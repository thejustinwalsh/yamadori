// Option instead of Result: invalid input returns undefined instead of throwing.
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = parseHexColor)]
pub fn parse_hex_color(s: &str) -> Option<Vec<u8>> {
    let digits = s.strip_prefix('#')?;
    if !digits.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let full: String = match digits.len() {
        3 => digits.chars().flat_map(|c| [c, c]).collect(),
        6 => digits.to_string(),
        _ => return None,
    };
    (0..3).map(|i| u8::from_str_radix(&full[2 * i..2 * i + 2], 16).ok()).collect()
}
