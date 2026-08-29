use crate::errors::{CompactError, Result};

/// Vector transform trait — `src/transform.rs:4`
pub trait Transform: Send + Sync {
    fn dims(&self) -> usize;
    fn transform(&self, input: &[f32], output: &mut [f32]) -> Result<()>;
    fn transform_vector(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.dims() {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims(),
                found: input.len(),
            });
        }
        let mut out = vec![0.0; self.dims()];
        self.transform(input, &mut out)?;
        Ok(out)
    }
}

/// No-op transform.
#[derive(Debug, Clone)]
pub struct Identity {
    dims: usize,
}

impl Identity {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Transform for Identity {
    fn dims(&self) -> usize {
        self.dims
    }
    fn transform(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.dims || output.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: input.len(),
            });
        }
        output.copy_from_slice(input);
        Ok(())
    }
}

/// Normalize to unit length (L2).
#[derive(Debug, Clone)]
pub struct Normalizer {
    dims: usize,
    epsilon: f32,
}

impl Normalizer {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            epsilon: 1e-12,
        }
    }
    pub fn with_epsilon(dims: usize, eps: f32) -> Self {
        Self { dims, epsilon: eps }
    }
}

impl Transform for Normalizer {
    fn dims(&self) -> usize {
        self.dims
    }
    fn transform(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.dims || output.len() != self.dims {
            return Err(CompactError::DimensionMismatch {
                expected: self.dims,
                found: input.len(),
            });
        }
        let mut sum = 0.0f32;
        for &x in input {
            sum += x * x;
        }
        let norm = sum.sqrt().max(self.epsilon);
        for (o, &i) in output.iter_mut().zip(input.iter()) {
            *o = i / norm;
        }
        Ok(())
    }
}

/// Center by subtracting per-dimension mean (computed at build time).
#[derive(Debug, Clone)]
pub struct Centering {
    mean: Vec<f32>,
}

impl Centering {
    pub fn from_data(data: &[Vec<f32>]) -> Result<Self> {
        if data.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        let dims = data[0].len();
        let mut mean = vec![0.0f32; dims];
        for v in data {
            if v.len() != dims {
                return Err(CompactError::DimensionMismatch {
                    expected: dims,
                    found: v.len(),
                });
            }
            for (d, &x) in v.iter().enumerate() {
                mean[d] += x;
            }
        }
        for m in &mut mean {
            *m /= data.len() as f32;
        }
        Ok(Self { mean })
    }

    pub fn new(mean: Vec<f32>) -> Result<Self> {
        if mean.is_empty() {
            return Err(CompactError::invalid_header("centering mean empty"));
        }
        Ok(Self { mean })
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }
}

impl Transform for Centering {
    fn dims(&self) -> usize {
        self.mean.len()
    }
    fn transform(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.mean.len() || output.len() != self.mean.len() {
            return Err(CompactError::DimensionMismatch {
                expected: self.mean.len(),
                found: input.len(),
            });
        }
        for (o, (&i, &m)) in output.iter_mut().zip(input.iter().zip(self.mean.iter())) {
            *o = i - m;
        }
        Ok(())
    }
}

/// Standardize: (x - mean) / stddev per dimension.
#[derive(Debug, Clone)]
pub struct Standardizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Standardizer {
    pub fn from_data(data: &[Vec<f32>]) -> Result<Self> {
        if data.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        let dims = data[0].len();
        let n = data.len() as f32;
        let mut mean = vec![0.0f32; dims];
        for v in data {
            if v.len() != dims {
                return Err(CompactError::DimensionMismatch {
                    expected: dims,
                    found: v.len(),
                });
            }
            for (d, &x) in v.iter().enumerate() {
                mean[d] += x;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0f32; dims];
        for v in data {
            for (d, &x) in v.iter().enumerate() {
                let diff = x - mean[d];
                var[d] += diff * diff;
            }
        }
        let mut std = vec![0.0f32; dims];
        for d in 0..dims {
            var[d] /= n;
            std[d] = var[d].sqrt().max(1e-6);
        }
        Ok(Self { mean, std })
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }
    pub fn std(&self) -> &[f32] {
        &self.std
    }
}

impl Transform for Standardizer {
    fn dims(&self) -> usize {
        self.mean.len()
    }
    fn transform(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.mean.len() || output.len() != self.mean.len() {
            return Err(CompactError::DimensionMismatch {
                expected: self.mean.len(),
                found: input.len(),
            });
        }
        for (o, ((&i, &m), &s)) in output
            .iter_mut()
            .zip(input.iter().zip(self.mean.iter()).zip(self.std.iter()))
        {
            *o = (i - m) / s;
        }
        Ok(())
    }
}

/// Chain two transforms sequentially.
pub struct Chain<A: Transform, B: Transform> {
    a: A,
    b: B,
}

impl<A: Transform, B: Transform> Chain<A, B> {
    pub fn new(a: A, b: B) -> Result<Self> {
        if a.dims() != b.dims() {
            return Err(CompactError::DimensionMismatch {
                expected: a.dims(),
                found: b.dims(),
            });
        }
        Ok(Self { a, b })
    }
}

impl<A: Transform, B: Transform> Transform for Chain<A, B> {
    fn dims(&self) -> usize {
        self.a.dims()
    }
    fn transform(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let mut tmp = vec![0.0; self.a.dims()];
        self.a.transform(input, &mut tmp)?;
        self.b.transform(&tmp, output)
    }
}

/// Apply transform to a whole dataset (allocates new Vecs).
pub fn transform_dataset<T: Transform>(t: &T, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(data.len());
    for v in data {
        out.push(t.transform_vector(v)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_unit() {
        let n = Normalizer::new(2);
        let mut out = vec![0.0; 2];
        n.transform(&[3.0, 4.0], &mut out).expect("norm");
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn centering() {
        let data = vec![vec![0.0, 10.0], vec![2.0, 20.0]];
        let c = Centering::from_data(&data).expect("center");
        assert_eq!(c.mean(), &[1.0, 15.0]);
        let mut out = vec![0.0; 2];
        c.transform(&[1.0, 15.0], &mut out).expect("t");
        assert!(out[0].abs() < 1e-6 && out[1].abs() < 1e-6);
    }

    #[test]
    fn standardizer() {
        let data = vec![vec![0.0], vec![2.0], vec![4.0]];
        let s = Standardizer::from_data(&data).expect("std");
        let mut out = vec![0.0; 1];
        s.transform(&[2.0], &mut out).expect("t");
        assert!(out[0].abs() < 1e-6); // mean is 2, so 2 -> 0
    }

    #[test]
    fn chain() {
        let n = Normalizer::new(2);
        let id = Identity::new(2);
        let chained = Chain::new(n.clone(), id).expect("chain");
        let mut out = vec![0.0; 2];
        chained.transform(&[3.0, 4.0], &mut out).expect("c");
        assert!((out[0] - 0.6).abs() < 1e-6);
    }
}
