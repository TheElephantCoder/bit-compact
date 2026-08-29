use crate::errors::{CompactError, Result};

/// Tiny dataset helpers — `src/dataset.rs:3`
/// No external crates; parses simple JSON-like arrays and generates synthetic data.

#[derive(Debug, Clone)]
pub struct Dataset {
    pub vectors: Vec<Vec<f32>>,
    pub dims: usize,
}

impl Dataset {
    pub fn new(vectors: Vec<Vec<f32>>) -> Result<Self> {
        if vectors.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        let dims = vectors[0].len();
        if dims == 0 {
            return Err(CompactError::invalid_header("dataset dims 0"));
        }
        for v in &vectors {
            if v.len() != dims {
                return Err(CompactError::DimensionMismatch {
                    expected: dims,
                    found: v.len(),
                });
            }
        }
        Ok(Self { vectors, dims })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Generate synthetic uniform vectors in [low, high).
    /// Deterministic via xorshift, no rand crate.
    pub fn synthetic_uniform(count: usize, dims: usize, low: f32, high: f32, seed: u64) -> Self {
        let mut vectors = Vec::with_capacity(count);
        let mut state = seed.max(1);
        for _ in 0..count {
            let mut v = Vec::with_capacity(dims);
            for _ in 0..dims {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state as f64 / u64::MAX as f64) as f32;
                v.push(low + u * (high - low));
            }
            vectors.push(v);
        }
        Self { vectors, dims }
    }

    /// Generate clustered data: `clusters` centers uniform, then gaussian noise around them
    pub fn synthetic_clustered(
        count: usize,
        dims: usize,
        clusters: usize,
        spread: f32,
        seed: u64,
    ) -> Self {
        let centers = Self::synthetic_uniform(clusters, dims, -1.0, 1.0, seed);
        let mut state = seed.wrapping_mul(0x9e3779b97f4a7c15);
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let center = &centers.vectors[i % clusters];
            let mut v = Vec::with_capacity(dims);
            for d in 0..dims {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let n = ((state % 2000) as f32 / 1000.0 - 1.0) * spread; // approx uniform noise, cheap
                v.push(center[d] + n);
            }
            out.push(v);
        }
        Self { vectors: out, dims }
    }

    /// Parse a simple JSON array of arrays: "[[0,1,2],[3,4,5]]"
    /// Accepts whitespace, no serde.
    pub fn from_json_array(s: &str) -> Result<Self> {
        let s = s.trim();
        if !s.starts_with('[') || !s.ends_with(']') {
            return Err(CompactError::invalid_header("json must start with ["));
        }
        // very small hand parser
        let mut vectors = Vec::new();
        let mut depth = 0usize;
        let mut cur_vec: Vec<f32> = Vec::new();
        let mut cur_num = String::new();
        let mut in_vec = false;

        for ch in s.chars() {
            match ch {
                '[' => {
                    depth += 1;
                    if depth == 2 {
                        cur_vec.clear();
                        in_vec = true;
                    }
                }
                ']' => {
                    if !cur_num.trim().is_empty() && in_vec {
                        let v: f32 = cur_num.trim().parse().map_err(|_| {
                            CompactError::invalid_header(format!("bad float {cur_num}"))
                        })?;
                        cur_vec.push(v);
                        cur_num.clear();
                    }
                    if depth == 2 && in_vec {
                        vectors.push(cur_vec.clone());
                        in_vec = false;
                    }
                    depth -= 1;
                }
                ',' => {
                    if in_vec && !cur_num.trim().is_empty() {
                        let v: f32 = cur_num.trim().parse().map_err(|_| {
                            CompactError::invalid_header(format!("bad float {cur_num}"))
                        })?;
                        cur_vec.push(v);
                        cur_num.clear();
                    }
                }
                c if c.is_whitespace() => {
                    // allow whitespace inside numbers? ignore
                }
                _ => {
                    cur_num.push(ch);
                }
            }
        }
        if vectors.is_empty() {
            return Err(CompactError::EmptyDataset);
        }
        Self::new(vectors)
    }

    pub fn to_vec(self) -> Vec<Vec<f32>> {
        self.vectors
    }

    pub fn sample(&self, n: usize) -> Vec<Vec<f32>> {
        self.vectors.iter().take(n).cloned().collect()
    }
}

/// Helpers for GUI: format vectors as comma strings
pub fn format_vector(v: &[f32], max: usize) -> String {
    let slice = if v.len() > max { &v[..max] } else { v };
    slice
        .iter()
        .map(|x| format!("{:.4}", x))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic() {
        let ds = Dataset::synthetic_uniform(10, 4, -1.0, 1.0, 42);
        assert_eq!(ds.len(), 10);
        assert_eq!(ds.dims, 4);
        assert!(ds
            .vectors
            .iter()
            .all(|v| v.iter().all(|&x| x >= -1.0 && x < 1.0)));
    }

    #[test]
    fn json_parse() {
        let ds = Dataset::from_json_array("[[0, 1, 2], [3,4,5]]").expect("parse");
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.vectors[0], vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn clustered() {
        let ds = Dataset::synthetic_clustered(20, 8, 3, 0.1, 123);
        assert_eq!(ds.len(), 20);
        assert_eq!(ds.dims, 8);
    }
}
