// Result<_, String>: wasm-bindgen throws the bare string, which is not an Error instance.
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = parseHexColor)]
pub fn parse_hex_color(s: &str) -> Result<Vec<u8>, String> {
    let bad = || format!("invalid hex color: {s}");
    let digits = s.strip_prefix('#').ok_or_else(bad)?;
    if !digits.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(bad());
    }
    let full: String = match digits.len() {
        3 => digits.chars().flat_map(|c| [c, c]).collect(),
        6 => digits.to_string(),
        _ => return Err(bad()),
    };
    (0..3)
        .map(|i| u8::from_str_radix(&full[2 * i..2 * i + 2], 16).map_err(|_| bad()))
        .collect()
}
