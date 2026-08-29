# bit-compact

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![Apache](https://img.shields.io/badge/apache-%23D42029.svg?style=for-the-badge&logo=apache&logoColor=white)

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
- **Mechanical sympathy**: 64B cache-line awareness (`aligned::CACHE_LINE`), 4096B disk block alignment (`aligned::DISK_BLOCK`), raw `&[u8]` zero-copy paths.

## Quantization Formulae

```text
quantized = floor(((v - min) / (max - min)) * 255)  clamped 0..=255
dequant   = min + (q / 255) * (max - min)
```
Per-dimension global `min`/`max` from calibration. Handles degenerate `max==min` and clamps out-of-range values.

Calibration variants:
- `Quantizer::calibrate` — per-dim min/max (tightest error)
- `Quantizer::calibrate_global` — single global range (uniform, memory-savvy)
- `Quantizer::calibrate_percentile(5.0, 95.0)` — outlier-robust clipping

Batch: `quantize_batch` / `dequantize_batch`, zero-alloc `quantize_into` / `dequantize_into`.

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

Builder config:

```rust
use bit_compact::{CompactConfig, QuantType, DistanceMetric};
let cfg = CompactConfig::builder(128, QuantType::SQ8, DistanceMetric::Cosine)
    .align_disk_blocks(true).build().unwrap();
```

## Features

### `config` — Tunables via `CompactConfig`/`WriterConfig`/`ReaderConfig`
Builder pattern, validation (`dims <= 65535`), version, `align_disk_blocks`, `verify_on_open`.

### `distance` — Optimized metrics
`l2_squared`, `l2`, `dot`, `cosine_distance`, `inner_product_distance`, `batch_distance`, `normalize`. 4-wide unrolled, dispatch via `DistanceMetric::distance`. Used by search and reader's `search()`.

### `stats` — Error & compression analysis
`evaluate(quantizer, dataset)` → `QuantizationReport { mse, mae, max_abs_error, snr_db, per_dim_mse, compression_ratio }`, helpers `theoretical_max_error`, `theoretical_mse_uniform`.

### `search` — Brute-force & parallel top-k
`brute_force_search(reader, query, k, distance_fn)`, `parallel_search(reader, query, k, threads, ...)`, `batch_search`, `ScanIter`. `CompactReader::search` / `search_parallel` / `iter()` convenience wrappers.

### `aligned` — Cache-line & block alignment
`CACHE_LINE=64`, `DISK_BLOCK=4096`, `AlignedBuffer<ALIGN>`, `CacheAlignedBuffer`, `BlockAlignedBuffer`, `StackBuf<N>` (repr(align(64))), `align_up`, `is_cache_aligned`.

### `storage` — Writer/Reader engine
`CompactWriter::{create, create_with_config, create_with_version, append, append_quantized, append_batch, finalize, finalize_with_padding, estimated_file_size}`, `CompactReader::{open, open_unverified, open_with_config, get_quantized_into, get_dequantized_into, get_vector, get_batch, search, search_parallel, iter, scan_quantized, verify_checksum, read_header}`. `Send + Sync` via `Arc<Mutex<File>>`.

### `cache` — Hot-vector LRU
`LruCache::new(capacity)`, `CachedReader::new(reader, cap)` — `get(id)` hits cache or does 1-seek then inserts, `hit_rate()`, `Arc`-shared for threads, evicts LRU on `capacity`.

### `batch` — Parallel ingestion
`BatchWriter` (buffer + `flush_to`), `parallel_calibrate(data, threads)`, `ChunkedReader::new(reader, batch_size)` iterator of `Vec<Vec<f32>>`, `parallel_batch_search` for many queries.

### `validate` — Integrity
`validate(path) -> ValidationReport {header, checksum_valid, row_ids_monotonic, metadata_finite, warnings}`, `quick_check(path)` (header+footer bounds only), summary `is_valid()`.

### `transform` — Vector pre-processing
`Transform` trait (`Identity`, `Normalizer`, `Centering::from_data`, `Standardizer::from_data`, `Chain::new`), `transform_dataset`.

### `header` / `sha`
`Header` 32B BE exact, `validate_footer`, `Sha256` FIPS 180-4 pure std.

### CLI — `bitcompact` binary
Zero-dep arg parser, subcommands `create`, `info`, `get`, `search`, `validate`, `stats`:

```bash
cargo run --bin bitcompact -- create vectors.btcp --dims 8 --metric cosine --align
cargo run --bin bitcompact -- info vectors.btcp
cargo run --bin bitcompact -- get vectors.btcp 42
cargo run --bin bitcompact -- search vectors.btcp --query 0.1,0.2,0.3 --k 5
cargo run --bin bitcompact -- validate vectors.btcp
```

## Examples

```bash
cargo run --example basic     # write + 1-seek read + batch + config
cargo run --example search    # top-k + parallel + batch + iter
cargo run --example stats     # report + global/percentile + aligned buf
cargo run --example cache     # LRU cache + hit rate
cargo run --example transform # center/standardize/normalize + chain
cargo run --release --bench quant_bench  # 10k x128 quantize + l2
```

## Crate Layout

- `src/errors.rs` — `CompactError` (`IoError`, `InvalidMagicBytes`, `DimensionMismatch`, `CorruptedFooter`, `QuantizationOverflow`, …)
- `src/quant.rs` — `Quantizer` (`calibrate`, `calibrate_global`, `calibrate_percentile`, `quantize_vector`, `dequantize_vector`, `*_into`, `*_batch`)
- `src/config.rs` — `CompactConfig` builder, `WriterConfig`, `ReaderConfig`
- `src/aligned.rs` — `AlignedBuffer`, `CACHE_LINE`, `DISK_BLOCK`, `StackBuf`
- `src/distance.rs` — L2/cosine/IP, 4-wide loops, batch
- `src/stats.rs` — `QuantizationReport`, `evaluate`, `snr_db`, `mse`
- `src/search.rs` — `brute_force_search`, `parallel_search`, `ScanIter`
- `src/cache.rs` — `LruCache`, `CachedReader` (hot-vector)
- `src/batch.rs` — `BatchWriter`, `ChunkedReader`, `parallel_calibrate`
- `src/validate.rs` — `validate`, `ValidationReport`, `quick_check`
- `src/transform.rs` — `Transform`, `Normalizer`, `Centering`, `Standardizer`, `Chain`
- `src/header.rs` — `Header` 32B BE
- `src/sha.rs` — `Sha256` FIPS
- `src/storage.rs` — `CompactWriter`, `CompactReader` (1-seek, `Send+Sync`)
- `src/bin/bitcompact.rs` — CLI (`create`/`info`/`get`/`search`/`validate`/`stats`)
- `src/lib.rs` — re-exports + `VERSION_MAJOR/MINOR`

## Build

```bash
cargo test
cargo test --release   # lto=true, codegen-units=1, panic=abort
cargo fmt --check
cargo clippy -- -D warnings
```

`profile.release` is tuned for analytical engines (`opt-level=3`, `lto=true`, `codegen-units=1`, `panic="abort"`).

## License

MIT OR Apache-2.0 — see `LICENSE-MIT` and `LICENSE-APACHE`.

## Repository

https://github.com/TheElephantCoder/bit-compact
