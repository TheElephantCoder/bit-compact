use crate::errors::{CompactError, Result};

/// Tiny vector ops — `src/ops.rs:3`
/// All ops are allocation-friendly and check dims.

pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(CompactError::DimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    Ok(())
}

pub fn sub(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(CompactError::DimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
    Ok(())
}

pub fn scale(a: &[f32], s: f32, out: &mut [f32]) -> Result<()> {
    if a.len() != out.len() {
        return Err(CompactError::DimensionMismatch {
            expected: a.len(),
            found: out.len(),
        });
    }
    for i in 0..a.len() {
        out[i] = a[i] * s;
    }
    Ok(())
}

pub fn l2_norm(a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= a.len() {
        sum += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3];
        i += 4;
    }
    while i < a.len() {
        sum += a[i] * a[i];
        i += 1;
    }
    sum.sqrt()
}

pub fn mean(vectors: &[Vec<f32>]) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        return Err(CompactError::EmptyDataset);
    }
    let dims = vectors[0].len();
    let mut out = vec![0.0; dims];
    for v in vectors {
        if v.len() != dims {
            return Err(CompactError::DimensionMismatch {
                expected: dims,
                found: v.len(),
            });
        }
        for d in 0..dims {
            out[d] += v[d];
        }
    }
    for x in &mut out {
        *x /= vectors.len() as f32;
    }
    Ok(out)
}

pub fn variance(vectors: &[Vec<f32>], mean: &[f32]) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        return Err(CompactError::EmptyDataset);
    }
    let dims = mean.len();
    let mut var = vec![0.0; dims];
    for v in vectors {
        if v.len() != dims {
            return Err(CompactError::DimensionMismatch {
                expected: dims,
                found: v.len(),
            });
        }
        for d in 0..dims {
            let diff = v[d] - mean[d];
            var[d] += diff * diff;
        }
    }
    for x in &mut var {
        *x /= vectors.len() as f32;
    }
    Ok(var)
}

pub fn clamp(v: &mut [f32], lo: f32, hi: f32) {
    for x in v {
        if *x < lo {
            *x = lo;
        } else if *x > hi {
            *x = hi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn add_sub() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let mut out = [0.0; 2];
        add(&a, &b, &mut out).unwrap();
        assert_eq!(out, [4.0, 6.0]);
        sub(&b, &a, &mut out).unwrap();
        assert_eq!(out, [2.0, 2.0]);
    }
    #[test]
    fn norm() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-6);
    }
}
