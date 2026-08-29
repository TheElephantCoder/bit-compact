# bit-compact

Zero-dependency low-level multimodal embedding compression & file format engine in Rust — SQ8 scalar quantization with O(1) random-access binary storage.

Inspired by columnar formats like Lance. Provides 4× space reduction (`f32` → `u8`), Big-Endian portable layout, zero-allocation seeks, and `Send + Sync` readers for concurrent analytical scans.

## Binary Layout (Big-Endian)

```
Header (32B): MAGIC "BTCP" | major u16 | minor u16 | dims u16 | quant_type u16 | distance u16 | reserved u16 | count u64 | footer_offset u64
Metadata (2*dims*4B): min_vec [f32; dims] BE | max_vec [f32; dims] BE
Data (count*dims*1B): quantized u8 stream
Footer (@footer_offset): row_ids [u64; count] BE | sha256 [u8;32] (data block)
```

All integers/floats are Big-Endian for disk/network portability. Header is 32B (2B reserved aligns `u64` fields to 8B and fills cache-line half). Optional 4096B alignment padding before footer (`CompactWriter::finalize_with_padding(true)`).

## Performance Goals

- **Zero-allocation seeks**: `CompactReader::get_quantized_into(index, &mut [u8])` does exactly 1 `seek` + 1 `read_exact`, no heap alloc for alignment.
- **4× reduction**: `f32` (4B) → `u8` (1B) per coordinate via linear SQ8.
- **Mechanical sympathy**: 64B cache-line awareness, 4096B disk block alignment, raw `&[u8]` zero-copy paths.

## Quantization Formulae

```text
quantized = floor(((v - min) / (max - min)) * 255)  clamped 0..=255
dequant   = min + (q / 255) * (max - min)
```
Per-dimension global `min`/`max` from calibration. Handles degenerate `max==min` and clamps out-of-range values.

## Quick Start

```rust
use bit_compact::{Quantizer, QuantType, DistanceMetric, CompactWriter, CompactReader};

let dataset = vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]];
let quantizer = Quantizer::calibrate(&dataset).unwrap();

let mut w = CompactWriter::create("vectors.btcp", quantizer, QuantType::SQ8, DistanceMetric::L2).unwrap();
for v in &dataset { w.append(v).unwrap(); }
w.finalize().unwrap();

let r = CompactReader::open("vectors.btcp").unwrap();
let mut buf = vec![0u8; r.dims()];
r.get_quantized_into(0, &mut buf).unwrap(); // 1 seek, 0 alloc
let dequant = r.get_vector(0).unwrap();
```

`CompactReader` is `Send + Sync` — share via `Arc<CompactReader>` for multi-threaded scans.

## Crate Layout

- `src/errors.rs` — `CompactError` (`IoError`, `InvalidMagicBytes`, `DimensionMismatch`, `CorruptedFooter`, `QuantizationOverflow`, …)
- `src/quant.rs` — `Quantizer::{quantize_vector,dequantize_vector,quantize_into,dequantize_into,calibrate}`
- `src/storage.rs` — `Header`, `Sha256` (FIPS 180-4, no deps), `CompactWriter`, `CompactReader`
- `src/lib.rs` — re-exports + `VERSION_MAJOR/MINOR`

## Build

```bash
cargo test
cargo test --release   # lto=true, codegen-units=1, panic=abort
```

`profile.release` is tuned for analytical engines (`opt-level=3`, `lto=true`, `codegen-units=1`, `panic="abort"`).

## License

MIT OR Apache-2.0

## Repository

https://github.com/TheElephantCoder/bit-compact
