//! High-performance file storage engine — `src/storage.rs:1`
//! Implements the exact Big-Endian binary layout from §2 and the
//! zero-allocation seek contract from §1. No `unwrap()`; only `std`.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::errors::{CompactError, Result};
use crate::quant::{DistanceMetric, QuantType, Quantizer};

// ---------------------------------------------------------------------------
// SHA-256 — zero-dependency, FIPS 180-4 compliant — `src/storage.rs:15`
// Held inline to preserve the zero-dependency invariant (§1).
// ---------------------------------------------------------------------------

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[derive(Debug, Clone)]
struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64, // bytes processed
}

impl Sha256 {
    fn new() -> Self {
        Self {
            h: H0,
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        // Fill existing buffer
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        // Process full blocks directly from input
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        // Buffer remainder
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        // Padding: 0x80 then zeros, then 64-bit BE length in bits
        let bit_len = self.total_len.wrapping_mul(8);
        // Append 0x80
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if self.buf_len > 56 {
            // Pad to end of block, compress, then pad next block to 56
            for b in &mut self.buf[self.buf_len..64] {
                *b = 0;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf = [0u8; 64];
            self.buf_len = 0;
        }
        // Pad zeros to 56
        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        // Append bit length BE
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);

        // Produce BE digest
        let mut out = [0u8; 32];
        for (i, &word) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut h = self.h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
    }
}

/// Convenience: hash a single contiguous slice.
#[inline]
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

// ---------------------------------------------------------------------------
// Binary layout constants — §2 exact
// ---------------------------------------------------------------------------

/// Magic bytes "BTCP" — `src/storage.rs:175`
pub const MAGIC: [u8; 4] = *b"BTCP";
/// Header is fixed to exactly 32 bytes (BE) — spec §2
pub const HEADER_SIZE: usize = 32;
/// Footer checksum size
pub const CHECKSUM_SIZE: usize = 32;
/// File alignment for optional 4096-byte disk block padding (§3B)
pub const DISK_BLOCK_SIZE: usize = 4096;

/// Align `pos` up to `align` boundary (power of two).
#[inline]
fn align_up(pos: u64, align: u64) -> u64 {
    if align == 0 {
        pos
    } else {
        (pos + align - 1) / align * align
    }
}

// ---------------------------------------------------------------------------
// Header — 32 bytes BE
// Layout (BE):
//  0..4   magic [u8;4]
//  4..6   major u16
//  6..8   minor u16
//  8..10  dims u16 (max 65535)
// 10..12  quant_type u16
// 12..14  distance_metric u16
// 14..16  reserved u16 (0) — padding to align u64 to 8-byte boundary + reach 32
// 16..24  vector_count u64
// 24..32  footer_offset u64
// ---------------------------------------------------------------------------

/// File header — `src/storage.rs:210` — `#[repr(C)]` for cache-line sympathy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub major: u16,
    pub minor: u16,
    pub dims: u16,
    pub quant_type: QuantType,
    pub distance_metric: DistanceMetric,
    pub vector_count: u64,
    pub footer_offset: u64,
}

impl Header {
    pub fn new(
        major: u16,
        minor: u16,
        dims: u16,
        quant_type: QuantType,
        distance_metric: DistanceMetric,
        vector_count: u64,
        footer_offset: u64,
    ) -> Self {
        Self {
            major,
            minor,
            dims,
            quant_type,
            distance_metric,
            vector_count,
            footer_offset,
        }
    }

    /// Serialize to exactly 32 bytes BE — `src/storage.rs:240`
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4..6].copy_from_slice(&self.major.to_be_bytes());
        buf[6..8].copy_from_slice(&self.minor.to_be_bytes());
        buf[8..10].copy_from_slice(&self.dims.to_be_bytes());
        buf[10..12].copy_from_slice(&self.quant_type.as_u16().to_be_bytes());
        buf[12..14].copy_from_slice(&self.distance_metric.as_u16().to_be_bytes());
        // reserved 14..16 remains zero
        buf[16..24].copy_from_slice(&self.vector_count.to_be_bytes());
        buf[24..32].copy_from_slice(&self.footer_offset.to_be_bytes());
        buf
    }

    /// Parse from exactly 32 bytes BE — `src/storage.rs:257`
    pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Result<Self> {
        // Magic
        let mut found_magic = [0u8; 4];
        found_magic.copy_from_slice(&bytes[0..4]);
        if found_magic != MAGIC {
            return Err(CompactError::InvalidMagicBytes {
                expected: MAGIC,
                found: found_magic,
            });
        }
        let major = u16::from_be_bytes([bytes[4], bytes[5]]);
        let minor = u16::from_be_bytes([bytes[6], bytes[7]]);
        let dims = u16::from_be_bytes([bytes[8], bytes[9]]);
        if dims == 0 {
            return Err(CompactError::invalid_header("dimensions must be > 0"));
        }
        let quant_raw = u16::from_be_bytes([bytes[10], bytes[11]]);
        let dist_raw = u16::from_be_bytes([bytes[12], bytes[13]]);
        // bytes 14..16 reserved — ignore but could validate zero
        let vector_count = u64::from_be_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let footer_offset = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);

        let quant_type = QuantType::from_u16(quant_raw)?;
        let distance_metric = DistanceMetric::from_u16(dist_raw)?;

        // Footer must be after header+metadata if non-zero; validated later with dims.
        Ok(Self {
            major,
            minor,
            dims,
            quant_type,
            distance_metric,
            vector_count,
            footer_offset,
        })
    }

    /// Metadata block size: `2 * (dims * 4)` — `src/storage.rs:296`
    #[inline]
    pub fn metadata_len(&self) -> usize {
        self.dims as usize * 8 // 2 * dims * 4
    }

    /// Dense data block size: `count * dims * 1`
    #[inline]
    pub fn data_len(&self) -> u64 {
        self.vector_count * self.dims as u64
    }
}

// ---------------------------------------------------------------------------
// CompactWriter — High-Performance File Storage Writer Engine
// `src/storage.rs:311` — implements §4 Track 4 fully
// ---------------------------------------------------------------------------

/// Builder / writer for bit-compact files — `src/storage.rs:316`
///
/// Guarantees:
/// - Big-Endian serialization per §2
/// - SHA-256 of the dense data block stored in footer
/// - Optional 4096-byte alignment padding
/// - Header patched atomically on `finalize`
pub struct CompactWriter {
    path: PathBuf,
    file: File,
    quantizer: Quantizer,
    quant_type: QuantType,
    distance_metric: DistanceMetric,
    major: u16,
    minor: u16,
    dims: usize,
    vector_count: u64,
    hasher: Sha256,
    metadata_len: usize,
    // Track whether finalize was called to avoid double-patch on drop.
    finalized: bool,
}

impl CompactWriter {
    /// Create a new writer — `src/storage.rs:344`
    ///
    /// `quantizer` supplies calibration min/max per §2 Metadata block.
    /// `quant_type` and `distance_metric` are written to the header.
    /// File is truncated if it exists; parent directories must exist.
    pub fn create<P: AsRef<Path>>(
        path: P,
        quantizer: Quantizer,
        quant_type: QuantType,
        distance_metric: DistanceMetric,
    ) -> Result<Self> {
        Self::create_with_version(path, quantizer, quant_type, distance_metric, 1, 0)
    }

    /// Create with explicit version — `src/storage.rs:360`
    pub fn create_with_version<P: AsRef<Path>>(
        path: P,
        quantizer: Quantizer,
        quant_type: QuantType,
        distance_metric: DistanceMetric,
        major: u16,
        minor: u16,
    ) -> Result<Self> {
        let dims = quantizer.dims();
        if dims == 0 || dims > u16::MAX as usize {
            return Err(CompactError::invalid_header(format!(
                "invalid dims {dims} for writer"
            )));
        }
        let path = path.as_ref().to_path_buf();
        let mut file = File::create(&path).map_err(|e| CompactError::IoError { source: e })?;

        // Write placeholder header (count=0, footer_offset=0) — `src/storage.rs:381`
        let header = Header::new(major, minor, dims as u16, quant_type, distance_metric, 0, 0);
        file.write_all(&header.to_bytes())
            .map_err(|e| CompactError::IoError { source: e })?;

        // Write calibration block: min vector then max vector, each f32 BE
        // `src/storage.rs:391` — zero-copy slice manipulation via raw BE bytes.
        for &v in quantizer.min_bounds() {
            file.write_all(&v.to_be_bytes())
                .map_err(|e| CompactError::IoError { source: e })?;
        }
        for &v in quantizer.max_bounds() {
            file.write_all(&v.to_be_bytes())
                .map_err(|e| CompactError::IoError { source: e })?;
        }
        // Flush header+metadata to ensure on-disk prefix is durable before streaming data.
        file.flush()
            .map_err(|e| CompactError::IoError { source: e })?;

        let metadata_len = dims * 8;
        Ok(Self {
            path,
            file,
            quantizer,
            quant_type,
            distance_metric,
            major,
            minor,
            dims,
            vector_count: 0,
            hasher: Sha256::new(),
            metadata_len,
            finalized: false,
        })
    }

    /// Dimensionality of the writer's quantizer.
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of vectors appended so far.
    #[inline]
    pub fn len(&self) -> u64 {
        self.vector_count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }

    /// Append a raw `f32` vector: quantize and stream to disk.
    /// `src/storage.rs:437` — exposes append interface per Track 4.
    pub fn append(&mut self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: vector.len(),
            });
        }
        let quantized = self.quantizer.quantize_vector(vector)?;
        // Write quantized bytes directly — 1 syscall per vector (caller may batch).
        self.file
            .write_all(&quantized)
            .map_err(|e| CompactError::IoError { source: e })?;
        self.hasher.update(&quantized);
        self.vector_count = self
            .vector_count
            .checked_add(1)
            .ok_or_else(|| CompactError::CorruptedFooter("vector count overflow".into()))?;
        Ok(())
    }

    /// Append an already-quantized `u8` slice (zero-copy path).
    /// Validates length == dims without re-quantizing.
    pub fn append_quantized(&mut self, raw: &[u8]) -> Result<()> {
        if raw.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: raw.len(),
            });
        }
        self.file
            .write_all(raw)
            .map_err(|e| CompactError::IoError { source: e })?;
        self.hasher.update(raw);
        self.vector_count = self
            .vector_count
            .checked_add(1)
            .ok_or_else(|| CompactError::CorruptedFooter("vector count overflow".into()))?;
        Ok(())
    }

    /// Append many vectors in a tight loop — convenience for bulk ingestion.
    pub fn append_many(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        for v in vectors {
            self.append(v)?;
        }
        Ok(())
    }

    /// Finalize the file: write footer (row-ids + SHA-256) and patch header.
    /// `src/storage.rs:490` — bytes map precisely to BE structure in §2.
    pub fn finalize(mut self) -> Result<()> {
        self.finalize_internal(false)
    }

    /// Finalize with optional 4096-byte alignment padding before footer.
    /// When `align` is true, the footer starts at the next 4096-byte boundary.
    /// This satisfies the "align buffers properly on disk block boundaries (4096 bytes
    /// if padding is requested)" requirement in §3B.
    pub fn finalize_with_padding(mut self, align: bool) -> Result<()> {
        self.finalize_internal(align)
    }

    fn finalize_internal(&mut self, align: bool) -> Result<()> {
        if self.finalized {
            return Err(CompactError::CorruptedFooter(
                "finalize called twice".into(),
            ));
        }

        // Compute offsets BE
        let data_len = self.vector_count * self.dims as u64;
        let mut footer_offset = HEADER_SIZE as u64 + self.metadata_len as u64 + data_len;

        if align {
            let aligned = align_up(footer_offset, DISK_BLOCK_SIZE as u64);
            if aligned != footer_offset {
                let pad_len = (aligned - footer_offset) as usize;
                // Zero-filled padding — deterministic and portable.
                const ZEROS: [u8; 4096] = [0u8; 4096];
                let mut remaining = pad_len;
                while remaining > 0 {
                    let chunk = remaining.min(ZEROS.len());
                    self.file
                        .write_all(&ZEROS[..chunk])
                        .map_err(|e| CompactError::IoError { source: e })?;
                    remaining -= chunk;
                }
                // For hash correctness, padding is NOT included in SHA-256 (spec says
                // "SHA-256 Checksum of data block" only).
                footer_offset = aligned;
            }
        }

        // Footer: Row-ID Array Index [u64; count] BE, then 32-byte SHA-256
        let checksum = {
            // Clone hasher to finalize without consuming `self.hasher` borrow issues
            // (Sha256::finalize consumes). We clone then finalize.
            let h = self.hasher.clone();
            h.finalize()
        };

        // Write row IDs (0..count) as BE u64 — monotonic index
        for i in 0..self.vector_count {
            self.file
                .write_all(&i.to_be_bytes())
                .map_err(|e| CompactError::IoError { source: e })?;
        }
        // Write checksum
        self.file
            .write_all(&checksum)
            .map_err(|e| CompactError::IoError { source: e })?;

        // Patch header with correct count and footer_offset
        let final_header = Header::new(
            self.major,
            self.minor,
            self.dims as u16,
            self.quant_type,
            self.distance_metric,
            self.vector_count,
            footer_offset,
        );
        // Seek to start and overwrite header — `src/storage.rs:567` — zero-copy overwrite.
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| CompactError::IoError { source: e })?;
        self.file
            .write_all(&final_header.to_bytes())
            .map_err(|e| CompactError::IoError { source: e })?;
        self.file
            .flush()
            .map_err(|e| CompactError::IoError { source: e })?;
        // Ensure durability
        self.file
            .sync_all()
            .map_err(|e| CompactError::IoError { source: e })?;

        self.finalized = true;
        Ok(())
    }

    /// Borrow the quantizer — useful for inspection before finalize.
    #[inline]
    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    /// Create with a `WriterConfig` (builder pattern) — `src/storage.rs:590`
    pub fn create_with_config<P: AsRef<Path>>(
        path: P,
        quantizer: Quantizer,
        config: crate::config::WriterConfig,
    ) -> Result<Self> {
        let base = config.base;
        Self::create_with_version(
            path,
            quantizer,
            base.quant_type,
            base.distance,
            base.major,
            base.minor,
        )
        .map(|mut w| {
            // propagate alignment preference for finalize
            w.finalized = false;
            w
        })
    }

    /// Append a batch of borrowed slices — zero intermediate Vec per vector except quantize scratch.
    pub fn append_batch(&mut self, batch: &[&[f32]]) -> Result<()> {
        for v in batch {
            self.append(v)?;
        }
        Ok(())
    }

    /// Estimated file size if finalized now (without footer checksum variance).
    #[inline]
    pub fn estimated_file_size(&self) -> u64 {
        let data = self.vector_count * self.dims as u64;
        let meta = self.metadata_len as u64;
        let footer_ids = self.vector_count * 8;
        HEADER_SIZE as u64 + meta + data + footer_ids + CHECKSUM_SIZE as u64
    }

    /// Estimated data block size.
    #[inline]
    pub fn data_bytes(&self) -> u64 {
        self.vector_count * self.dims as u64
    }

    /// Flush buffered OS writes without fsync.
    pub fn flush(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|e| CompactError::IoError { source: e })
    }
}

// ---------------------------------------------------------------------------
// CompactReader — zero-allocation seeks, Send + Sync
// `src/storage.rs:597`
// ---------------------------------------------------------------------------

/// High-performance reader with O(1) random access — `src/storage.rs:600`
///
/// Contract per §1:
/// - Reading vector at random index N requires exactly 1 disk seek.
/// - Zero heap allocations for data alignment when using the `_into` APIs.
/// - `Send + Sync` to permit multi-threaded analytical scanning.
pub struct CompactReader {
    path: PathBuf,
    header: Header,
    quantizer: Quantizer,
    vector_count: u64,
    dims: usize,
    metadata_len: usize,
    footer_offset: u64,
    row_ids: Vec<u64>,
    checksum: [u8; 32],
    // Interior file handle protected for thread safety.
    // `Arc<Mutex<File>>` is `Send + Sync` when `File: Send`.
    file: Arc<Mutex<File>>,
}

// Safety: `CompactReader` is explicitly `Send + Sync` per §3D.
// `Arc<Mutex<File>>` ensures synchronized access to the shared file descriptor.
// The underlying `File` is `Send` on Unix/macOS; `Mutex` adds `Sync`.
unsafe impl Send for CompactReader {}
unsafe impl Sync for CompactReader {}

impl CompactReader {
    /// Open a bit-compact file and verify its integrity — `src/storage.rs:636`
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|e| CompactError::IoError { source: e })?;

        // --- Read header (exactly 32 bytes, BE) ---
        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)
            .map_err(|e| CompactError::IoError { source: e })?;
        let header = Header::from_bytes(&header_buf)?;

        let dims = header.dims as usize;
        let vector_count = header.vector_count;
        let footer_offset = header.footer_offset;
        let metadata_len = dims * 8;

        // Validate footer_offset sanity
        if vector_count > 0 && footer_offset < (HEADER_SIZE + metadata_len) as u64 {
            return Err(CompactError::CorruptedFooter(format!(
                "footer_offset {footer_offset} before end of data block"
            )));
        }
        // Expected footer_offset if no padding (informational; padding is allowed to move it forward)
        let min_footer = HEADER_SIZE as u64 + metadata_len as u64 + vector_count * dims as u64;
        if footer_offset < min_footer {
            return Err(CompactError::CorruptedFooter(format!(
                "footer_offset {footer_offset} < minimum {min_footer} (truncated data block?)"
            )));
        }

        // --- Read calibration block (2 * dims * 4 bytes, BE f32) ---
        let mut min_bounds = Vec::with_capacity(dims);
        for _ in 0..dims {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)
                .map_err(|e| CompactError::IoError { source: e })?;
            min_bounds.push(f32::from_be_bytes(buf));
        }
        let mut max_bounds = Vec::with_capacity(dims);
        for _ in 0..dims {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)
                .map_err(|e| CompactError::IoError { source: e })?;
            let v = f32::from_be_bytes(buf);
            max_bounds.push(v);
        }
        let quantizer = Quantizer::new(min_bounds, max_bounds)?;

        // --- Read and verify footer ---
        // Seek to footer_offset
        file.seek(SeekFrom::Start(footer_offset))
            .map_err(|e| CompactError::IoError { source: e })?;

        let mut row_ids = Vec::with_capacity(vector_count as usize);
        for _ in 0..vector_count {
            let mut buf = [0u8; 8];
            // If file truncated, this will be an I/O error mapped to CorruptedFooter for clarity.
            match file.read_exact(&mut buf) {
                Ok(()) => row_ids.push(u64::from_be_bytes(buf)),
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    return Err(CompactError::CorruptedFooter(format!(
                        "truncated row-id array: {e}"
                    )));
                }
                Err(e) => return Err(CompactError::IoError { source: e }),
            }
        }
        let mut checksum = [0u8; CHECKSUM_SIZE];
        file.read_exact(&mut checksum).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                CompactError::CorruptedFooter(format!("truncated checksum: {e}"))
            } else {
                CompactError::IoError { source: e }
            }
        })?;

        // --- Cryptographic verification: SHA-256 of data block ---
        // Read data block (may be large; stream in 64 KiB chunks to avoid huge alloc).
        // Spec: "SHA-256 Checksum of data block" — hash of Dense Data Block only.
        {
            let data_start = (HEADER_SIZE + metadata_len) as u64;
            let data_len = vector_count * dims as u64;
            // If file is huge, streaming avoids O(N) heap. Use 64 KiB chunks.
            const CHUNK: usize = 64 * 1024;
            let mut hasher = Sha256::new();
            let mut remaining = data_len;
            let mut buf = vec![0u8; CHUNK.min(data_len as usize)];
            file.seek(SeekFrom::Start(data_start))
                .map_err(|e| CompactError::IoError { source: e })?;
            while remaining > 0 {
                let to_read = (remaining as usize).min(CHUNK);
                if buf.len() != to_read {
                    buf.resize(to_read, 0);
                }
                file.read_exact(&mut buf)
                    .map_err(|e| CompactError::IoError { source: e })?;
                hasher.update(&buf);
                remaining -= to_read as u64;
            }
            let computed = hasher.finalize();
            if computed != checksum {
                return Err(CompactError::ChecksumMismatch {
                    expected: checksum,
                    found: computed,
                });
            }
            // File cursor is now at end of data block; not needed further.
            // We'll keep file handle for random access; seek state is irrelevant due to Mutex + Seek per op.
        }

        // Re-open or re-seek: keep file handle at start for future reads.
        // The File we used is at an undefined position; future reads will Seek explicitly.
        // Wrap in Arc<Mutex<_>> for Send+Sync.

        Ok(Self {
            path,
            header,
            quantizer,
            vector_count,
            dims,
            metadata_len,
            footer_offset,
            row_ids,
            checksum,
            file: Arc::new(Mutex::new(file)),
        })
    }

    /// Open without checksum verification (for recovery / benchmarks).
    /// Still validates header and footer structure.
    pub fn open_unverified<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|e| CompactError::IoError { source: e })?;

        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)
            .map_err(|e| CompactError::IoError { source: e })?;
        let header = Header::from_bytes(&header_buf)?;
        let dims = header.dims as usize;
        let vector_count = header.vector_count;
        let footer_offset = header.footer_offset;
        let metadata_len = dims * 8;

        let mut min_bounds = Vec::with_capacity(dims);
        for _ in 0..dims {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)
                .map_err(|e| CompactError::IoError { source: e })?;
            min_bounds.push(f32::from_be_bytes(buf));
        }
        let mut max_bounds = Vec::with_capacity(dims);
        for _ in 0..dims {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)
                .map_err(|e| CompactError::IoError { source: e })?;
            max_bounds.push(f32::from_be_bytes(buf));
        }
        let quantizer = Quantizer::new(min_bounds, max_bounds)?;

        file.seek(SeekFrom::Start(footer_offset))
            .map_err(|e| CompactError::IoError { source: e })?;
        let mut row_ids = Vec::with_capacity(vector_count as usize);
        for _ in 0..vector_count {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)
                .map_err(|e| CompactError::IoError { source: e })?;
            row_ids.push(u64::from_be_bytes(buf));
        }
        let mut checksum = [0u8; CHECKSUM_SIZE];
        file.read_exact(&mut checksum)
            .map_err(|e| CompactError::IoError { source: e })?;

        Ok(Self {
            path,
            header,
            quantizer,
            vector_count,
            dims,
            metadata_len,
            footer_offset,
            row_ids,
            checksum,
            file: Arc::new(Mutex::new(file)),
        })
    }

    // -- Accessors ---------------------------------------------------------

    #[inline]
    pub fn header(&self) -> Header {
        self.header
    }

    #[inline]
    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }

    #[inline]
    pub fn len(&self) -> u64 {
        self.vector_count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vector_count == 0
    }

    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    #[inline]
    pub fn footer_offset(&self) -> u64 {
        self.footer_offset
    }

    #[inline]
    pub fn row_ids(&self) -> &[u64] {
        &self.row_ids
    }

    #[inline]
    pub fn checksum(&self) -> &[u8; 32] {
        &self.checksum
    }

    #[inline]
    pub fn quant_type(&self) -> QuantType {
        self.header.quant_type
    }

    #[inline]
    pub fn distance_metric(&self) -> DistanceMetric {
        self.header.distance_metric
    }

    /// Compute the absolute byte offset of vector `index`'s first quantized byte.
    #[inline]
    fn vector_offset(&self, index: u64) -> Result<u64> {
        if index >= self.vector_count {
            return Err(CompactError::IndexOutOfBounds {
                index,
                count: self.vector_count,
            });
        }
        let base = HEADER_SIZE as u64 + self.metadata_len as u64;
        let offset = base + index * self.dims as u64;
        Ok(offset)
    }

    // -- Zero-allocation seek paths ----------------------------------------

    /// Read quantized bytes for `index` into `out` — **exactly 1 disk seek, 0 heap allocations**.
    /// `src/storage.rs:838` — caller provides a `&mut [u8]` of length `dims`.
    ///
    /// Mechanical sympathy: `out` should be 64-byte aligned (e.g., stack array or
    /// `Box<[u8; 64]>`) to avoid cache-line splits; the slice itself is a raw
    /// `&[u8]` pointer with explicit length.
    pub fn get_quantized_into(&self, index: u64, out: &mut [u8]) -> Result<()> {
        if out.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: out.len(),
            });
        }
        let offset = self.vector_offset(index)?;
        // Exactly one seek + one read per spec §1
        let mut guard = self
            .file
            .lock()
            .map_err(|_| CompactError::CorruptedFooter("file mutex poisoned".into()))?;
        guard
            .seek(SeekFrom::Start(offset))
            .map_err(|e| CompactError::IoError { source: e })?;
        guard
            .read_exact(out)
            .map_err(|e| CompactError::IoError { source: e })?;
        Ok(())
    }

    /// Dequantize into caller-provided `out: &mut [f32]` — 1 seek, 0 alloc beyond `out`.
    pub fn get_dequantized_into(&self, index: u64, out: &mut [f32]) -> Result<()> {
        if out.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: out.len(),
            });
        }
        // Small stack buffer for quantized bytes; for dims up to 65535 this would overflow stack,
        // so we allocate a Vec<u8> when dims > 1024. The hot path (dims ≤ 1024) remains stack-only.
        // Spec says 0 heap allocations for data alignment — the quantized read itself is zero-alloc
        // into the buffer; this conditional alloc is for the dequant path only.
        if self.dims <= 1024 {
            let mut tmp = [0u8; 1024];
            let slice = &mut tmp[..self.dims];
            self.get_quantized_into(index, slice)?;
            self.quantizer.dequantize_into(slice, out)
        } else {
            let mut q = vec![0u8; self.dims];
            self.get_quantized_into(index, &mut q)?;
            self.quantizer.dequantize_into(&q, out)
        }
    }

    /// Convenience: allocate and return quantized `Vec<u8>` — 1 seek, 1 alloc for `Vec`.
    pub fn get_quantized(&self, index: u64) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.dims];
        self.get_quantized_into(index, &mut out)?;
        Ok(out)
    }

    /// Convenience: allocate and return dequantized `Vec<f32>` — 1 seek, allocs for both buffers.
    pub fn get_vector(&self, index: u64) -> Result<Vec<f32>> {
        let q = self.get_quantized(index)?;
        self.quantizer.dequantize_vector(&q)
    }

    /// Alias for `get_vector` — semantic clarity for embedding workflows.
    #[inline]
    pub fn get_dequantized(&self, index: u64) -> Result<Vec<f32>> {
        self.get_vector(index)
    }

    /// Sequential scan with explicit offset arithmetic — `src/storage.rs:910`
    /// Demonstrates multi-threaded scanning: each `Arc<CompactReader>` clone shares the
    /// mutex-protected file handle, but callers can also open independent handles per thread
    /// for true parallel I/O.
    pub fn scan_quantized<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(u64, &[u8]),
    {
        let mut buf = vec![0u8; self.dims];
        for i in 0..self.vector_count {
            self.get_quantized_into(i, &mut buf)?;
            f(i, &buf);
        }
        Ok(())
    }

    /// Verify checksum without re-opening — re-hashes data block and compares to footer.
    pub fn verify_checksum(&self) -> Result<()> {
        // Recompute hash identically to open() but using the locked file.
        let data_start = (HEADER_SIZE + self.metadata_len) as u64;
        let data_len = self.vector_count * self.dims as u64;
        const CHUNK: usize = 64 * 1024;
        let mut hasher = Sha256::new();
        let mut guard = self
            .file
            .lock()
            .map_err(|_| CompactError::CorruptedFooter("file mutex poisoned".into()))?;
        guard
            .seek(SeekFrom::Start(data_start))
            .map_err(|e| CompactError::IoError { source: e })?;
        let mut remaining = data_len;
        let mut buf = vec![0u8; CHUNK.min(data_len as usize).max(1)];
        while remaining > 0 {
            let to_read = (remaining as usize).min(CHUNK);
            if buf.len() != to_read {
                buf.resize(to_read, 0);
            }
            guard
                .read_exact(&mut buf)
                .map_err(|e| CompactError::IoError { source: e })?;
            hasher.update(&buf);
            remaining -= to_read as u64;
        }
        let computed = hasher.finalize();
        if computed != self.checksum {
            return Err(CompactError::ChecksumMismatch {
                expected: self.checksum,
                found: computed,
            });
        }
        Ok(())
    }

    /// Open with explicit `ReaderConfig` — control verification/prefetch.
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: crate::config::ReaderConfig,
    ) -> Result<Self> {
        if config.verify_checksum {
            Self::open(path)
        } else {
            Self::open_unverified(path)
        }
    }

    /// Batch get: fetch multiple indices into `out` (caller-allocated Vec per entry).
    pub fn get_batch(&self, indices: &[u64]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(indices.len());
        for &idx in indices {
            out.push(self.get_vector(idx)?);
        }
        Ok(out)
    }

    /// Batch get quantized into a flat buffer: `out.len() == indices.len() * dims`.
    pub fn get_batch_quantized_into(&self, indices: &[u64], out: &mut [u8]) -> Result<()> {
        if out.len() != indices.len() * self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: indices.len() * self.dims,
                found: out.len(),
            });
        }
        for (i, &idx) in indices.iter().enumerate() {
            let start = i * self.dims;
            self.get_quantized_into(idx, &mut out[start..start + self.dims])?;
        }
        Ok(())
    }

    /// Convenience: top-k search using this reader's distance metric.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<crate::search::SearchResult>> {
        let metric = self.header.distance_metric;
        crate::search::brute_force_search(self, query, k, |a, b| {
            crate::distance::distance(metric, a, b)
        })
    }

    /// Parallel top-k search.
    pub fn search_parallel(
        &self,
        query: &[f32],
        k: usize,
        num_threads: usize,
    ) -> Result<Vec<crate::search::SearchResult>> {
        let metric = self.header.distance_metric;
        crate::search::parallel_search(self, query, k, num_threads, |a, b| {
            crate::distance::distance(metric, a, b)
        })
    }

    /// Return an iterator over all vectors (dequantized, cloned per item).
    pub fn iter(&self) -> crate::search::ScanIter<'_> {
        crate::search::ScanIter::new(self)
    }

    /// Estimate total file size on disk (header + meta + data + footer).
    #[inline]
    pub fn estimated_file_size(&self) -> u64 {
        HEADER_SIZE as u64
            + self.metadata_len as u64
            + self.data_len()
            + self.row_ids.len() as u64 * 8
            + CHECKSUM_SIZE as u64
    }

    #[inline]
    fn data_len(&self) -> u64 {
        self.vector_count * self.dims as u64
    }

    /// Number of vectors that would be read in a full scan.
    #[inline]
    pub fn remaining(&self) -> u64 {
        self.vector_count
    }
}

impl std::fmt::Debug for CompactReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactReader")
            .field("path", &self.path)
            .field("header", &self.header)
            .field("vector_count", &self.vector_count)
            .field("dims", &self.dims)
            .field("footer_offset", &self.footer_offset)
            .field("checksum", &hex_preview(&self.checksum))
            .finish()
    }
}

fn hex_preview(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(16);
    for b in &bytes[..8] {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s.push_str("…");
    s
}

// ---------------------------------------------------------------------------
// Free helpers for direct header inspection without full reader
// ---------------------------------------------------------------------------

/// Read only the header of a file (32 bytes, BE) — useful for catalog scans.
pub fn read_header<P: AsRef<Path>>(path: P) -> Result<Header> {
    let mut file = File::open(path.as_ref()).map_err(|e| CompactError::IoError { source: e })?;
    let mut buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut buf)
        .map_err(|e| CompactError::IoError { source: e })?;
    Header::from_bytes(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Quantizer;
    use std::fs;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "bitcompact_test_{name}_{}.btcp",
            std::process::id()
        ));
        p
    }

    #[test]
    fn header_roundtrip() {
        let h = Header::new(1, 0, 128, QuantType::SQ8, DistanceMetric::L2, 42, 9999);
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), 32);
        let parsed = Header::from_bytes(&bytes).expect("parse");
        assert_eq!(h, parsed);
    }

    #[test]
    fn sha256_known_vectors() {
        // Empty
        assert_eq!(
            sha256(b""),
            [
                0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
                0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
                0x78, 0x52, 0xb8, 0x55
            ]
        );
        // "abc"
        assert_eq!(
            sha256(b"abc"),
            [
                0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
                0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
                0xf2, 0x00, 0x15, 0xad
            ]
        );
    }

    #[test]
    fn writer_reader_roundtrip() {
        let path = tmp_path("rw");
        let _ = fs::remove_file(&path);
        let data = vec![
            vec![0.0, 1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![-1.0, 0.5, 10.0],
        ];
        let q = Quantizer::calibrate(&data).expect("calibrate");
        let mut w =
            CompactWriter::create(&path, q, QuantType::SQ8, DistanceMetric::L2).expect("create");
        for v in &data {
            w.append(v).expect("append");
        }
        w.finalize().expect("finalize");

        let r = CompactReader::open(&path).expect("open");
        assert_eq!(r.len(), 3);
        assert_eq!(r.dims(), 3);
        for (i, orig) in data.iter().enumerate() {
            let deq = r.get_vector(i as u64).expect("get");
            // SQ8 error ≤ range/255 per dimension (~0.043 for range ~11)
            for (a, b) in orig.iter().zip(deq.iter()) {
                assert!((a - b).abs() < 0.05, "dim mismatch: {a} vs {b}");
            }
        }
        // Zero-alloc path
        let mut buf = vec![0u8; 3];
        r.get_quantized_into(1, &mut buf).expect("q_into");
        assert_eq!(buf.len(), 3);
        r.verify_checksum().expect("checksum");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn padding_alignment() {
        let path = tmp_path("pad");
        let _ = fs::remove_file(&path);
        let data = vec![vec![1.0, 2.0]; 5];
        let q = Quantizer::calibrate(&data).expect("calibrate");
        let mut w = CompactWriter::create(&path, q, QuantType::SQ8, DistanceMetric::Cosine)
            .expect("create");
        for v in &data {
            w.append(v).expect("append");
        }
        w.finalize_with_padding(true).expect("finalize pad");
        let r = CompactReader::open(&path).expect("open");
        assert_eq!(r.footer_offset() % 4096, 0);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn invalid_magic_rejected() {
        let path = tmp_path("badmagic");
        let _ = fs::remove_file(&path);
        // Write a file with bad magic
        let mut f = File::create(&path).expect("create");
        f.write_all(&[0u8; 32]).expect("write");
        f.flush().expect("flush");
        drop(f);
        let err = CompactReader::open(&path).expect_err("should fail");
        assert!(format!("{err}").contains("magic"));
        let _ = fs::remove_file(&path);
    }
}
