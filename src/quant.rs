//! SQ8 compression core — `src/quant.rs:1`
//! Scalar quantization mapping `f32 -> u8` per the exact formulae in §3C.
//! No `unwrap()` is used; all fallible paths return `CompactError`.

use crate::errors::{CompactError, Result};

// ---------------------------------------------------------------------------
// Enumerations matching the binary layout spec §2 (u16 tags, BE)
// ---------------------------------------------------------------------------

/// Quantization type tag — `src/quant.rs:12`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum QuantType {
    RawF32 = 0,
    SQ8 = 1,
    ProductQ = 2,
}

impl QuantType {
    #[inline]
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    pub fn from_u16(v: u16) -> Result<Self> {
        match v {
            0 => Ok(Self::RawF32),
            1 => Ok(Self::SQ8),
            2 => Ok(Self::ProductQ),
            other => Err(CompactError::InvalidQuantType(other)),
        }
    }
}

/// Distance metric tag — `src/quant.rs:43`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum DistanceMetric {
    L2 = 0,
    Cosine = 1,
    InnerProduct = 2,
}

impl DistanceMetric {
    #[inline]
    pub fn as_u16(self) -> u16 {
        self as u16
    }

    pub fn from_u16(v: u16) -> Result<Self> {
        match v {
            0 => Ok(Self::L2),
            1 => Ok(Self::Cosine),
            2 => Ok(Self::InnerProduct),
            other => Err(CompactError::InvalidDistanceMetric(other)),
        }
    }
}

// ---------------------------------------------------------------------------
// Quantizer — per-dimension linear SQ8 mapping
// ---------------------------------------------------------------------------

/// Holds the evaluated dimensional vector extents (min/max bounds).
/// `src/quant.rs:78` — aligns to cache-line commentary §1 / §3B (64 bytes).
#[derive(Debug, Clone)]
pub struct Quantizer {
    /// Lower bounds per dimension — length == `dims`.
    min_bounds: Vec<f32>,
    /// Upper bounds per dimension — length == `dims`.
    max_bounds: Vec<f32>,
    /// Dimensionality (cached for fast asserts).
    dims: usize,
}

// Safety: `Quantizer` contains only `Vec<f32>` (Send + Sync) so it is
// automatically `Send + Sync`. Assertions below guarantee the trait bounds.
#[allow(dead_code)]
fn _assert_send_sync()
where
    Quantizer: Send + Sync,
{
}

impl Quantizer {
    /// Construct from explicit per-dimension extents — `src/quant.rs:101`
    ///
    /// Validates:
    /// - `min_bounds.len() == max_bounds.len() > 0`
    /// - `dims <= 65535` (u16::MAX per header)
    /// - no NaN, `max >= min` per dimension
    pub fn new(min_bounds: Vec<f32>, max_bounds: Vec<f32>) -> Result<Self> {
        if min_bounds.len() != max_bounds.len() {
            return Err(CompactError::DimensionMismatch {
                expected: min_bounds.len(),
                found: max_bounds.len(),
            });
        }
        let dims = min_bounds.len();
        if dims == 0 {
            return Err(CompactError::invalid_header(
                "quantizer dimensions must be > 0",
            ));
        }
        if dims > u16::MAX as usize {
            return Err(CompactError::invalid_header(format!(
                "dimensions {dims} exceeds u16::MAX (65535)"
            )));
        }
        for i in 0..dims {
            let min = min_bounds[i];
            let max = max_bounds[i];
            if !min.is_finite() || !max.is_finite() {
                return Err(CompactError::QuantizationOverflow {
                    dimension: i,
                    value: if !min.is_finite() { min } else { max },
                    min,
                    max,
                    reason: "non-finite bound",
                });
            }
            if max < min {
                return Err(CompactError::invalid_header(format!(
                    "max < min at dim {i}: max {max} < min {min}"
                )));
            }
        }
        Ok(Self {
            min_bounds,
            max_bounds,
            dims,
        })
    }

    /// Calibrate from a full dataset: compute per-dimension global min/max.
    /// `src/quant.rs:152` — O(N*D) scan, no allocation beyond bounds.
    pub fn calibrate(vectors: &[Vec<f32>]) -> Result<Self> {
        if vectors.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        let dims = vectors[0].len();
        if dims == 0 {
            return Err(CompactError::invalid_header(
                "calibration vectors must have dimensionality > 0",
            ));
        }
        if dims > u16::MAX as usize {
            return Err(CompactError::invalid_header(format!(
                "dimensions {dims} exceeds u16::MAX"
            )));
        }
        // Validate uniform dimensionality without unwrap.
        for (idx, v) in vectors.iter().enumerate() {
            if v.len() != dims {
                return Err(CompactError::DimensionMismatch {
                    expected: dims,
                    found: v.len(),
                });
            }
            // Early NaN/Inf detection yields clearer errors at calibration.
            for (d, &val) in v.iter().enumerate() {
                if !val.is_finite() {
                    return Err(CompactError::QuantizationOverflow {
                        dimension: d,
                        value: val,
                        min: f32::NEG_INFINITY,
                        max: f32::INFINITY,
                        reason: "non-finite value during calibration",
                    });
                }
            }
            // Suppress unused warning for idx in non-error path.
            let _ = idx;
        }

        let mut min_bounds = vec![f32::INFINITY; dims];
        let mut max_bounds = vec![f32::NEG_INFINITY; dims];

        for vec in vectors {
            for (d, &val) in vec.iter().enumerate() {
                if val < min_bounds[d] {
                    min_bounds[d] = val;
                }
                if val > max_bounds[d] {
                    max_bounds[d] = val;
                }
            }
        }

        // If any dimension remained INF (should not happen with non-empty data).
        for d in 0..dims {
            if !min_bounds[d].is_finite() || !max_bounds[d].is_finite() {
                return Err(CompactError::CorruptedData(format!(
                    "failed to calibrate dimension {d}"
                )));
            }
        }

        Ok(Self {
            min_bounds,
            max_bounds,
            dims,
        })
    }

    /// Calibrate from a slice of slices (zero-copy view) — `src/quant.rs:230`
    pub fn calibrate_slices(vectors: &[&[f32]]) -> Result<Self> {
        if vectors.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        let dims = vectors[0].len();
        if dims == 0 {
            return Err(CompactError::invalid_header(
                "calibration slices must have dimensionality > 0",
            ));
        }
        if dims > u16::MAX as usize {
            return Err(CompactError::invalid_header(format!(
                "dimensions {dims} exceeds u16::MAX"
            )));
        }
        for v in vectors {
            if v.len() != dims {
                return Err(CompactError::DimensionMismatch {
                    expected: dims,
                    found: v.len(),
                });
            }
            for (d, &val) in v.iter().enumerate() {
                if !val.is_finite() {
                    return Err(CompactError::QuantizationOverflow {
                        dimension: d,
                        value: val,
                        min: f32::NEG_INFINITY,
                        max: f32::INFINITY,
                        reason: "non-finite value during calibration",
                    });
                }
            }
        }

        let mut min_bounds = vec![f32::INFINITY; dims];
        let mut max_bounds = vec![f32::NEG_INFINITY; dims];
        for vec in vectors {
            for (d, &val) in vec.iter().enumerate() {
                if val < min_bounds[d] {
                    min_bounds[d] = val;
                }
                if val > max_bounds[d] {
                    max_bounds[d] = val;
                }
            }
        }
        Ok(Self {
            min_bounds,
            max_bounds,
            dims,
        })
    }

    /// Dimensionality — `src/quant.rs:290`
    #[inline]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Borrow lower bounds — zero-copy accessor.
    #[inline]
    pub fn min_bounds(&self) -> &[f32] {
        &self.min_bounds
    }

    /// Borrow upper bounds — zero-copy accessor.
    #[inline]
    pub fn max_bounds(&self) -> &[f32] {
        &self.max_bounds
    }

    /// Global calibration: single min/max across all dimensions (memory-savvy, uniform range).
    pub fn calibrate_global(vectors: &[Vec<f32>]) -> Result<Self> {
        if vectors.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        let dims = vectors[0].len();
        if dims == 0 {
            return Err(CompactError::invalid_header(
                "global calibrate: dims must be > 0",
            ));
        }
        let mut gmin = f32::INFINITY;
        let mut gmax = f32::NEG_INFINITY;
        for v in vectors {
            if v.len() != dims {
                return Err(CompactError::DimensionMismatch {
                    expected: dims,
                    found: v.len(),
                });
            }
            for &x in v {
                if !x.is_finite() {
                    return Err(CompactError::QuantizationOverflow {
                        dimension: 0,
                        value: x,
                        min: gmin,
                        max: gmax,
                        reason: "non-finite in global calibrate",
                    });
                }
                if x < gmin {
                    gmin = x;
                }
                if x > gmax {
                    gmax = x;
                }
            }
        }
        Ok(Self {
            min_bounds: vec![gmin; dims],
            max_bounds: vec![gmax; dims],
            dims,
        })
    }

    /// Robust calibration via percentile clipping (outlier resistant).
    /// `low` and `high` are percentiles in [0,100], e.g., 1.0 and 99.0 clip 1% tails.
    pub fn calibrate_percentile(vectors: &[Vec<f32>], low: f32, high: f32) -> Result<Self> {
        if vectors.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        if !(0.0..=100.0).contains(&low) || !(0.0..=100.0).contains(&high) || low >= high {
            return Err(CompactError::invalid_header(format!(
                "invalid percentiles low={low} high={high}"
            )));
        }
        let dims = vectors[0].len();
        let base = Self::calibrate(vectors)?;
        // For each dim, collect values, sort, pick percentiles.
        // Stable, std-only, O(D * N log N).
        let n = vectors.len();
        let mut mins = Vec::with_capacity(dims);
        let mut maxs = Vec::with_capacity(dims);
        for d in 0..dims {
            let mut vals: Vec<f32> = vectors.iter().map(|v| v[d]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo_idx = ((low / 100.0) * (n as f32 - 1.0)).floor() as usize;
            let hi_idx = ((high / 100.0) * (n as f32 - 1.0)).ceil() as usize;
            let lo = vals[lo_idx.min(n - 1)];
            let hi = vals[hi_idx.min(n - 1)];
            // Ensure we never collapse to zero range; expand by epsilon if needed.
            let (mn, mx) = if (hi - lo).abs() < 1e-6 {
                (base.min_bounds[d], base.max_bounds[d])
            } else {
                (lo, hi)
            };
            mins.push(mn);
            maxs.push(mx);
        }
        Self::new(mins, maxs)
    }

    /// Batch quantize — caller can pre-allocate `out` as `Vec<Vec<u8>>`.
    pub fn quantize_batch(&self, batch: &[Vec<f32>]) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(batch.len());
        for v in batch {
            out.push(self.quantize_vector(v)?);
        }
        Ok(out)
    }

    /// Batch dequantize.
    pub fn dequantize_batch(&self, batch: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(batch.len());
        for v in batch {
            out.push(self.dequantize_vector(v)?);
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Core transforms — formulae §3C verbatim
    // ------------------------------------------------------------------

    /// Quantize a single `f32` vector into packed `u8` bins.
    /// `src/quant.rs:314`
    ///
    /// Formula (spec §3C):
    /// `quantized = floor(((f32_val - min) / (max - min)) * 255.0)` clamped `0..=255`.
    ///
    /// Explicit iterator bounds mapping with per-dimension global extents.
    pub fn quantize_vector(&self, vec: &[f32]) -> Result<Vec<u8>> {
        if vec.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: vec.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dims);
        for (d, (&val, (&min, &max))) in vec
            .iter()
            .zip(self.min_bounds.iter().zip(self.max_bounds.iter()))
            .enumerate()
        {
            if !val.is_finite() {
                return Err(CompactError::QuantizationOverflow {
                    dimension: d,
                    value: val,
                    min,
                    max,
                    reason: "non-finite input value",
                });
            }
            let range = max - min;
            let quantized: u8 = if range.abs() < f32::EPSILON {
                // Degenerate dimension: all training values identical.
                // Clamp to 0 if at/near min, 255 if above (defensive).
                if (val - min).abs() < 1e-6 {
                    0u8
                } else if val < min {
                    0u8
                } else if val > max {
                    255u8
                } else {
                    0u8
                }
            } else {
                // Linear mapping with explicit clamp before floor.
                let mut normalized = (val - min) / range;
                // Clamp securely per spec — handle out-of-range eval vectors.
                if normalized < 0.0 {
                    normalized = 0.0;
                } else if normalized > 1.0 {
                    normalized = 1.0;
                }
                // floor(normalized * 255) — clamp ensures 0..=255 after conversion.
                let scaled = (normalized * 255.0).floor();
                // Defensive clamp for floating-point edge cases (255.0 epsilon).
                if scaled < 0.0 {
                    0u8
                } else if scaled > 255.0 {
                    255u8
                } else {
                    scaled as u8
                }
            };
            out.push(quantized);
        }
        Ok(out)
    }

    /// Dequantize packed `u8` bins back to approximate `f32` coordinates.
    /// `src/quant.rs:385`
    ///
    /// Formula (spec §3C):
    /// `f32_val = min + ((quantized as f32 / 255.0) * (max - min))`
    pub fn dequantize_vector(&self, raw: &[u8]) -> Result<Vec<f32>> {
        if raw.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: raw.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dims);
        for (d, (&byte, (&min, &max))) in raw
            .iter()
            .zip(self.min_bounds.iter().zip(self.max_bounds.iter()))
            .enumerate()
        {
            let _ = d; // used only if we add overflow detection later
            let range = max - min;
            let val = if range.abs() < f32::EPSILON {
                // Degenerate: reconstruct exact bound.
                min
            } else {
                min + ((byte as f32 / 255.0) * range)
            };
            // Guard against non-finite reconstruction (should be impossible).
            if !val.is_finite() {
                return Err(CompactError::QuantizationOverflow {
                    dimension: d,
                    value: val,
                    min,
                    max,
                    reason: "non-finite dequantized value",
                });
            }
            out.push(val);
        }
        Ok(out)
    }

    /// Zero-allocation quantized write into a caller-provided `out` buffer.
    /// Returns an error if `out.len() != dims` or `vec.len() != dims`.
    /// This is the mechanical-sympathy path: caller reuses a 64-byte aligned buffer.
    #[inline]
    pub fn quantize_into(&self, vec: &[f32], out: &mut [u8]) -> Result<()> {
        if vec.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: vec.len(),
            });
        }
        if out.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: out.len(),
            });
        }
        for (d, (&val, (&min, &max))) in vec
            .iter()
            .zip(self.min_bounds.iter().zip(self.max_bounds.iter()))
            .enumerate()
        {
            if !val.is_finite() {
                return Err(CompactError::QuantizationOverflow {
                    dimension: d,
                    value: val,
                    min,
                    max,
                    reason: "non-finite input value",
                });
            }
            let range = max - min;
            let q = if range.abs() < f32::EPSILON {
                0u8
            } else {
                let mut n = (val - min) / range;
                if n < 0.0 {
                    n = 0.0;
                } else if n > 1.0 {
                    n = 1.0;
                }
                let s = (n * 255.0).floor();
                if s < 0.0 {
                    0
                } else if s > 255.0 {
                    255
                } else {
                    s as u8
                }
            };
            out[d] = q;
        }
        Ok(())
    }

    /// Zero-allocation dequantize into caller-provided `out: &mut [f32]`.
    #[inline]
    pub fn dequantize_into(&self, raw: &[u8], out: &mut [f32]) -> Result<()> {
        if raw.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: raw.len(),
            });
        }
        if out.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: out.len(),
            });
        }
        for (d, (&byte, (&min, &max))) in raw
            .iter()
            .zip(self.min_bounds.iter().zip(self.max_bounds.iter()))
            .enumerate()
        {
            let range = max - min;
            let val = if range.abs() < f32::EPSILON {
                min
            } else {
                min + ((byte as f32 / 255.0) * range)
            };
            if !val.is_finite() {
                return Err(CompactError::QuantizationOverflow {
                    dimension: d,
                    value: val,
                    min,
                    max,
                    reason: "non-finite dequantized value",
                });
            }
            out[d] = val;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic() {
        let q = Quantizer::new(vec![0.0, -1.0], vec![1.0, 1.0]).expect("new");
        let v = vec![0.5, 0.0];
        let quantized = q.quantize_vector(&v).expect("quantize");
        // For dim 0: (0.5-0)/(1)*255 =127.5 floor=127
        // For dim 1: (0-(-1))/2*255=127.5 floor 127
        assert_eq!(quantized, vec![127, 127]);
        let deq = q.dequantize_vector(&quantized).expect("dequantize");
        // Approx error <= range/255
        assert!((deq[0] - 0.498).abs() < 0.01);
        assert!((deq[1] - (-0.0039)).abs() < 0.01);
    }

    #[test]
    fn clamp_out_of_range() {
        let q = Quantizer::new(vec![0.0], vec![1.0]).expect("new");
        // value above max should clamp to 255
        let high = q.quantize_vector(&[2.0]).expect("q");
        assert_eq!(high[0], 255);
        let low = q.quantize_vector(&[-1.0]).expect("q");
        assert_eq!(low[0], 0);
    }

    #[test]
    fn degenerate_range() {
        let q = Quantizer::new(vec![5.0], vec![5.0]).expect("new");
        let out = q.quantize_vector(&[5.0]).expect("q");
        assert_eq!(out[0], 0);
        let deq = q.dequantize_vector(&out).expect("dq");
        assert_eq!(deq[0], 5.0);
    }

    #[test]
    fn calibrate_simple() {
        let data = vec![vec![0.0, 10.0], vec![1.0, 20.0], vec![-1.0, 15.0]];
        let q = Quantizer::calibrate(&data).expect("calibrate");
        assert_eq!(q.min_bounds(), &[-1.0, 10.0]);
        assert_eq!(q.max_bounds(), &[1.0, 20.0]);
    }
}
