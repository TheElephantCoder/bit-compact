use crate::errors::Result;
use crate::quant::Quantizer;
use crate::transform::Transform;

/// A simple pipeline: transform → quantize → store
/// `src/pipeline.rs:5`
pub struct Pipeline {
    transforms: Vec<Box<dyn Transform>>,
    quantizer: Option<Quantizer>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { transforms: Vec::new(), quantizer: None }
    }

    pub fn with_quantizer(mut self, q: Quantizer) -> Self {
        self.quantizer = Some(q);
        self
    }

    pub fn add_transform<T: Transform + 'static>(mut self, t: T) -> Self {
        self.transforms.push(Box::new(t));
        self
    }

    pub fn set_quantizer(&mut self, q: Quantizer) {
        self.quantizer = Some(q);
    }

    /// Run pipeline on a single vector: apply transforms sequentially, then quantize/dequantize if quantizer present
    pub fn run(&self, mut vec: Vec<f32>) -> Result<Vec<f32>> {
        let dims = vec.len();
        for t in &self.transforms {
            if t.dims() != dims {
                return Err(crate::errors::CompactError::DimensionMismatch { expected: t.dims(), found: dims });
            }
            let mut out = vec![0.0; dims];
            t.transform(&vec, &mut out)?;
            vec = out;
        }
        if let Some(q) = &self.quantizer {
            let quant = q.quantize_vector(&vec)?;
            vec = q.dequantize_vector(&quant)?;
        }
        Ok(vec)
    }

    pub fn run_batch(&self, batch: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        batch.iter().map(|v| self.run(v.clone())).collect()
    }
}

impl Default for Pipeline {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::Normalizer;
    #[test]
    fn pipeline_normalize() {
        let p = Pipeline::new().add_transform(Normalizer::new(2));
        let out = p.run(vec![3.0,4.0]).unwrap();
        assert!((out[0]-0.6).abs()<1e-6);
    }
}
