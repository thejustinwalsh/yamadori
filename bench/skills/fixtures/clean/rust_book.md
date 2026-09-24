# Rust

Prefer `&str` over `&String` in function parameters.

```rust
fn greet(name: &str) -> String { format!("hi {name}") }
```

Run `cargo clippy` in CI.
