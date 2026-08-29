use crate::errors::{CompactError, Result};
use crate::quant::DistanceMetric;

/// Pure f32 distance helpers — `src/distance.rs:5`
/// All functions validate dimensionality and return `DimensionMismatch` if needed.
/// Loops are 4-wide unrolled to aid auto-vectorization without unsafe.

#[inline]
fn check_dims(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(CompactError::DimensionMismatch {
            expected: a.len(),
            found: b.len(),
        });
    }
    if a.is_empty() {
        return Err(CompactError::invalid_header("distance: dims must be > 0"));
    }
    Ok(())
}

/// Squared L2 (Euclidean) distance — faster when sqrt is unnecessary for ranking.
pub fn l2_squared(a: &[f32], b: &[f32]) -> Result<f32> {
    check_dims(a, b)?;
    let mut sum = 0.0f32;
    let mut i = 0;
    let n = a.len();
    // 4-wide unrolled
    while i + 4 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    Ok(sum)
}

pub fn l2(a: &[f32], b: &[f32]) -> Result<f32> {
    Ok(l2_squared(a, b)?.sqrt())
}

pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    check_dims(a, b)?;
    let mut sum = 0.0f32;
    let mut i = 0;
    let n = a.len();
    while i + 4 <= n {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    Ok(sum)
}

pub fn l2_norm(a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    let n = a.len();
    while i + 4 <= n {
        sum += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3];
        i += 4;
    }
    while i < n {
        sum += a[i] * a[i];
        i += 1;
    }
    sum.sqrt()
}

/// Cosine distance = 1 - cosine_similarity. Range [0, 2]. 0 = identical direction.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    check_dims(a, b)?;
    let mut dot_sum = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let mut i = 0;
    let n = a.len();
    while i + 4 <= n {
        dot_sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        na += a[i] * a[i] + a[i + 1] * a[i + 1] + a[i + 2] * a[i + 2] + a[i + 3] * a[i + 3];
        nb += b[i] * b[i] + b[i + 1] * b[i + 1] + b[i + 2] * b[i + 2] + b[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        dot_sum += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
        i += 1;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    let cos = (dot_sum / denom).clamp(-1.0, 1.0);
    Ok(1.0 - cos)
}

/// Cosine similarity in [-1, 1].
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    Ok(1.0 - cosine_distance(a, b)?)
}

/// Inner-product distance for maximum-inner-product search.
/// Defined as `-dot(a,b)` so smaller distance = larger inner product (ranking-friendly).
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    Ok(-dot(a, b)?)
}

/// Dispatch via `DistanceMetric` tag stored in header.
pub fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> Result<f32> {
    match metric {
        DistanceMetric::L2 => l2_squared(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::InnerProduct => inner_product_distance(a, b),
    }
}

/// Batch variant: compute distances from `query` to each `dataset` entry.
pub fn batch_l2_squared(query: &[f32], dataset: &[Vec<f32>]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(dataset.len());
    for v in dataset {
        out.push(l2_squared(query, v)?);
    }
    Ok(out)
}

pub fn batch_distance(
    metric: DistanceMetric,
    query: &[f32],
    dataset: &[Vec<f32>],
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(dataset.len());
    for v in dataset {
        out.push(distance(metric, query, v)?);
    }
    Ok(out)
}

/// Normalize in-place to unit length (for cosine workflows).
pub fn normalize(v: &mut [f32]) {
    let n = l2_norm(v);
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

pub fn normalized(a: &[f32]) -> Vec<f32> {
    let mut out = a.to_vec();
    normalize(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_squared(&a, &b).unwrap() - 25.0).abs() < 1e-6);
        assert!((l2(&a, &b).unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn dot_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b).unwrap() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!(cosine_distance(&a, &b).unwrap().abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_distance(&a, &b).unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dispatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        assert!(distance(DistanceMetric::L2, &a, &b).unwrap().abs() < 1e-6);
        assert!(distance(DistanceMetric::Cosine, &a, &b).unwrap().abs() < 1e-6);
        assert!(distance(DistanceMetric::InnerProduct, &a, &b).unwrap() < 0.0);
    }

    #[test]
    fn dim_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(l2(&a, &b).is_err());
    }
}
