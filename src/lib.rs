//! bit-compact — low-level multimodal embedding compression & file format engine
//! `src/lib.rs:1`
//!
//! Production-grade, zero-dependency data infrastructure for scalar-quantized
//! vector storage with O(1) random access.
//!
//! Binary layout is Big-Endian as specified in §2, with `Send + Sync` readers
//! for concurrent analytical scanning and a mechanically-sympathetic 64-byte
//! cache-line awareness.
//!
//! # Example
//!
//! ```no_run
//! use bit_compact::{CompactWriter, CompactReader, Quantizer, QuantType, DistanceMetric};
//!
//! let dataset = vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]];
//! let quantizer = Quantizer::calibrate(&dataset).expect("calibrate");
//! let mut writer = CompactWriter::create("embeddings.btcp", quantizer, QuantType::SQ8, DistanceMetric::L2).expect("create");
//! for vec in &dataset { writer.append(vec).expect("append"); }
//! writer.finalize().expect("finalize");
//!
//! let reader = CompactReader::open("embeddings.btcp").expect("open");
//! let mut buf = vec![0u8; reader.dims()];
//! reader.get_quantized_into(0, &mut buf).expect("seek"); // 1 seek, 0 alloc for alignment
//! let deq = reader.get_vector(0).expect("dequant");
//! ```

#![allow(clippy::many_single_char_names)]
#![allow(missing_docs)]

pub mod aligned;
pub mod batch;
pub mod cache;
pub mod config;
pub mod dataset;
pub mod distance;
pub mod errors;
pub mod header;
pub mod metrics;
pub mod ops;
pub mod quant;
pub mod search;
pub mod server;
pub mod sha;
pub mod stats;
pub mod storage;
pub mod transform;
pub mod validate;

// Re-exports for ergonomic crate root
pub use aligned::{
    AlignedBuffer, BlockAlignedBuffer, CacheAlignedBuffer, StackBuf, CACHE_LINE, DISK_BLOCK,
};
pub use batch::{parallel_batch_search, parallel_calibrate, BatchWriter, ChunkedReader};
pub use cache::{CachedReader, LruCache};
pub use config::{CompactConfig, ConfigBuilder, ReaderConfig, WriterConfig};
pub use dataset::{format_vector, Dataset};
pub use distance::{
    cosine_distance, cosine_similarity, dot, inner_product_distance, l2, l2_squared,
};
pub use errors::{CompactError, Result};
pub use header::{Header, DISK_BLOCK_SIZE, HEADER_SIZE, MAGIC};
pub use metrics::{evaluate_search, mrr, recall_precision_at_k, SearchMetrics};
pub use ops::{add as vec_add, l2_norm, mean as vec_mean, scale as vec_scale, sub as vec_sub};
pub use quant::{DistanceMetric, QuantType, Quantizer};
pub use search::{brute_force_search, parallel_search, SearchResult};
pub use sha::{sha256, Sha256};
pub use stats::{evaluate as evaluate_quantization, QuantizationReport};
pub use storage::{CompactReader, CompactWriter, CHECKSUM_SIZE};
pub use transform::{
    transform_dataset, Centering, Chain, Identity, Normalizer, Standardizer, Transform,
};
pub use validate::{quick_check, validate, ValidationReport};

// Crate version constants mirroring header version fields
/// Default major version written to new files.
pub const VERSION_MAJOR: u16 = 1;
/// Default minor version written to new files.
pub const VERSION_MINOR: u16 = 0;

// Compile-time assertions for mechanical sympathy invariants — `src/lib.rs:53`
const _: () = {
    // Header must be exactly 32 bytes per §2
    assert!(HEADER_SIZE == 32, "header must be 32 bytes");
    // Cache line is 64 bytes; header fits in half a line, metadata is contiguous.
    assert!(DISK_BLOCK_SIZE == 4096, "disk block must be 4096");
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::fs;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "bitcompact_integ_{name}_{}.btcp",
            std::process::id()
        ));
        p
    }

    #[test]
    fn end_to_end_4x_reduction() {
        // Demonstrate 4× space reduction: f32 (4 bytes) -> u8 (1 byte)
        let dims = 128usize;
        let count = 100u64;
        // Synthetic embeddings in [-1, 1]
        let dataset: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                (0..dims)
                    .map(|d| ((i * dims as u64 + d as u64) % 200) as f32 / 100.0 - 1.0)
                    .collect()
            })
            .collect();

        let f32_bytes = count as usize * dims * 4;
        let quantizer = Quantizer::calibrate(&dataset).expect("calibrate");
        let path = tmp("reduction");
        let _ = fs::remove_file(&path);
        let mut w = CompactWriter::create(&path, quantizer, QuantType::SQ8, DistanceMetric::Cosine)
            .expect("create");
        for v in &dataset {
            w.append(v).expect("append");
        }
        w.finalize().expect("finalize");

        let meta = fs::metadata(&path).expect("meta");
        let file_bytes = meta.len() as usize;
        // File bytes ≈ 32 header + 2*dims*4 metadata + count*dims*1 data + count*8 footer + 32 checksum
        let expected_data = count as usize * dims; // quantized payload
        let overhead = 32 + 2 * dims * 4 + count as usize * 8 + 32;
        assert_eq!(file_bytes, expected_data + overhead);
        // 4× reduction on the payload itself
        assert_eq!(f32_bytes / expected_data, 4);

        let reader = CompactReader::open(&path).expect("open");
        assert_eq!(reader.len(), count);
        // Random access spot-check
        let v = reader.get_vector(42).expect("get");
        assert_eq!(v.len(), dims);
        let _ = fs::remove_file(&path);
    }
}
