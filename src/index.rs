use crate::errors::Result;
use crate::storage::CompactReader;
use std::collections::HashMap;

/// Very small inverted index for demo — `src/index.rs:3`
/// Maps quantized byte value per dimension to list of ids (naive, for GUI demo)
#[derive(Debug, Default)]
pub struct SimpleIndex {
    dims: usize,
    // per dim, per byte value -> ids
    postings: Vec<HashMap<u8, Vec<u64>>>,
}

impl SimpleIndex {
    pub fn build(reader: &CompactReader) -> Result<Self> {
        let dims = reader.dims();
        let mut postings: Vec<HashMap<u8, Vec<u64>>> = vec![HashMap::new(); dims];
        let mut buf = vec![0u8; dims];
        for id in 0..reader.len() {
            reader.get_quantized_into(id, &mut buf)?;
            for (d, &b) in buf.iter().enumerate() {
                postings[d].entry(b).or_default().push(id);
            }
        }
        Ok(Self { dims, postings })
    }

    pub fn dims(&self) -> usize { self.dims }

    pub fn lookup(&self, dim: usize, byte: u8) -> Option<&[u64]> {
        self.postings.get(dim)?.get(&byte).map(|v| v.as_slice())
    }

    pub fn stats(&self) -> String {
        let total: usize = self.postings.iter().map(|m| m.values().map(|v| v.len()).sum::<usize>()).sum();
        format!("dims={} total_postings={} avg_per_dim={:.1}", self.dims, total, total as f64 / self.dims as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{Quantizer, QuantType, DistanceMetric};
    use crate::storage::CompactWriter;
    #[test]
    fn index_build() {
        let data=vec![vec![0.0,0.0],vec![1.0,1.0],vec![0.5,0.5]];
        let q=Quantizer::calibrate(&data).unwrap();
        let mut p=std::env::temp_dir(); p.push(format!("idx_{}.btcp", std::process::id()));
        let _=std::fs::remove_file(&p);
        let mut w=CompactWriter::create(&p,q,QuantType::SQ8,DistanceMetric::L2).unwrap();
        for v in &data { w.append(v).unwrap(); }
        w.finalize().unwrap();
        let r=CompactReader::open(&p).unwrap();
        let idx=SimpleIndex::build(&r).unwrap();
        assert_eq!(idx.dims(),2);
        let _=std::fs::remove_file(p);
    }
}
