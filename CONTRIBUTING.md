# Contributing

## Dev setup
```bash
cargo test
cargo test --release
cargo run --example basic
```

## Style
- `cargo fmt` before commit
- no `unwrap()` in library code, use `Result` + `CompactError`
- all on-disk integers/floats are Big-Endian (`to_be_bytes`)
- keep `Send + Sync` for `CompactReader`

## Adding features
- new quantization types: add variant to `QuantType` and header tag, update `Header`
- new distance: add to `distance.rs` and `DistanceMetric::distance`
- file format changes: bump `VERSION_MAJOR`/`MINOR` and add migration in `storage.rs`

## Testing
- add unit tests in module `#[cfg(test)]`
- add integration test in `src/lib.rs` or `tests/`
- verify 32B header `cargo test header::tests::roundtrip`

## Release
- bump `version` in `Cargo.toml`
- ensure `cargo test --release` passes with `lto=true codegen-units=1 panic=abort`
