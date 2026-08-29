use crate::quant::Quantizer;

/// Quantization error statistics — `src/stats.rs:4`
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationReport {
    pub dims: usize,
    pub num_vectors: usize,
    pub mse: f64,
    pub mae: f64,
    pub max_abs_error: f32,
    pub snr_db: f64,
    pub per_dim_mse: Vec<f64>,
    pub per_dim_range: Vec<f32>,
    pub compression_ratio: f64,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
}

impl QuantizationReport {
    pub fn summary(&self) -> String {
        format!(
            "dims={} n={} mse={:.6} mae={:.6} max_err={:.6} snr={:.2}dB ratio={:.2}x ({}B -> {}B)",
            self.dims,
            self.num_vectors,
            self.mse,
            self.mae,
            self.max_abs_error,
            self.snr_db,
            self.compression_ratio,
            self.original_bytes,
            self.compressed_bytes
        )
    }
}

/// Mean squared error between original and dequantized.
pub fn mse(original: &[f32], dequantized: &[f32]) -> f32 {
    assert_eq!(original.len(), dequantized.len());
    if original.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for (a, b) in original.iter().zip(dequantized.iter()) {
        let d = *a as f64 - *b as f64;
        sum += d * d;
    }
    (sum / original.len() as f64) as f32
}

pub fn mae(original: &[f32], dequantized: &[f32]) -> f32 {
    assert_eq!(original.len(), dequantized.len());
    if original.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for (a, b) in original.iter().zip(dequantized.iter()) {
        sum += (*a as f64 - *b as f64).abs();
    }
    (sum / original.len() as f64) as f32
}

pub fn max_abs_error(original: &[f32], dequantized: &[f32]) -> f32 {
    original
        .iter()
        .zip(dequantized.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |m, v| m.max(v))
}

/// Signal-to-noise ratio in dB: 10*log10(signal_power / noise_power)
pub fn snr_db(original: &[f32], dequantized: &[f32]) -> f64 {
    assert_eq!(original.len(), dequantized.len());
    if original.is_empty() {
        return f64::INFINITY;
    }
    let mut sig_pow = 0.0f64;
    let mut noise_pow = 0.0f64;
    for (a, b) in original.iter().zip(dequantized.iter()) {
        sig_pow += (*a as f64) * (*a as f64);
        let n = *a as f64 - *b as f64;
        noise_pow += n * n;
    }
    sig_pow /= original.len() as f64;
    noise_pow /= original.len() as f64;
    if noise_pow < 1e-12 {
        return f64::INFINITY;
    }
    if sig_pow < 1e-12 {
        return 0.0;
    }
    10.0 * (sig_pow / noise_pow).log10()
}

pub fn compression_ratio(original_bytes: usize, compressed_bytes: usize) -> f64 {
    if compressed_bytes == 0 {
        return f64::INFINITY;
    }
    original_bytes as f64 / compressed_bytes as f64
}

/// Build a full report for a dataset given its quantizer.
pub fn evaluate(quantizer: &Quantizer, dataset: &[Vec<f32>]) -> QuantizationReport {
    let dims = quantizer.dims();
    let n = dataset.len();
    if n == 0 || dims == 0 {
        return QuantizationReport {
            dims,
            num_vectors: n,
            mse: 0.0,
            mae: 0.0,
            max_abs_error: 0.0,
            snr_db: f64::INFINITY,
            per_dim_mse: vec![0.0; dims],
            per_dim_range: quantizer
                .max_bounds()
                .iter()
                .zip(quantizer.min_bounds().iter())
                .map(|(mx, mn)| mx - mn)
                .collect(),
            compression_ratio: 0.0,
            original_bytes: 0,
            compressed_bytes: 0,
        };
    }

    let mut total_mse = 0.0f64;
    let mut total_mae = 0.0f64;
    let mut max_err = 0.0f32;
    let mut per_dim_mse = vec![0.0f64; dims];
    let mut sig_pow = 0.0f64;
    let mut noise_pow = 0.0f64;

    let original_bytes = n * dims * 4;
    let compressed_bytes = n * dims; // u8

    for vec in dataset {
        let q = quantizer.quantize_vector(vec).expect("quantize in stats");
        let dq = quantizer.dequantize_vector(&q).expect("dequant in stats");
        for (d, (a, b)) in vec.iter().zip(dq.iter()).enumerate() {
            let diff = *a as f64 - *b as f64;
            let sq = diff * diff;
            total_mse += sq;
            total_mae += diff.abs();
            per_dim_mse[d] += sq;
            let ae = (a - b).abs();
            if ae > max_err {
                max_err = ae;
            }
            sig_pow += (*a as f64) * (*a as f64);
            noise_pow += sq;
        }
    }

    let total_elems = (n * dims) as f64;
    let mse_val = total_mse / total_elems;
    let mae_val = total_mae / total_elems;
    for v in &mut per_dim_mse {
        *v /= n as f64;
    }
    let snr = if noise_pow < 1e-12 {
        f64::INFINITY
    } else {
        sig_pow /= total_elems;
        noise_pow /= total_elems;
        if sig_pow < 1e-12 {
            0.0
        } else {
            10.0 * (sig_pow / noise_pow).log10()
        }
    };

    QuantizationReport {
        dims,
        num_vectors: n,
        mse: mse_val,
        mae: mae_val,
        max_abs_error: max_err,
        snr_db: snr,
        per_dim_mse,
        per_dim_range: quantizer
            .max_bounds()
            .iter()
            .zip(quantizer.min_bounds().iter())
            .map(|(mx, mn)| mx - mn)
            .collect(),
        compression_ratio: compression_ratio(original_bytes, compressed_bytes),
        original_bytes,
        compressed_bytes,
    }
}

/// Theoretical max error per dimension for SQ8: range / 255 / 2 (midpoint) or range/255 (floor).
pub fn theoretical_max_error(range: f32) -> f32 {
    range / 255.0
}

pub fn theoretical_mse_uniform(range: f32) -> f64 {
    // Variance of uniform quantization error in [-step/2, step/2] = step^2 / 12
    let step = range as f64 / 255.0;
    step * step / 12.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Quantizer;

    #[test]
    fn mse_zero_for_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(mse(&a, &b).abs() < 1e-6);
        assert!(snr_db(&a, &b).is_infinite());
    }

    #[test]
    fn mae_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert!((mae(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn evaluate_report() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.5, 0.5]];
        let q = Quantizer::calibrate(&data).expect("calibrate");
        let rep = evaluate(&q, &data);
        assert_eq!(rep.dims, 2);
        assert_eq!(rep.num_vectors, 3);
        assert!(rep.mse < 0.01);
        assert!((rep.compression_ratio - 4.0).abs() < 1e-6);
        assert!(rep.snr_db > 20.0);
    }

    #[test]
    fn compression_ratio_4x() {
        assert!((compression_ratio(400, 100) - 4.0).abs() < 1e-9);
    }
}
