use crate::errors::{CompactError, Result};
use crate::quant::Quantizer;
use crate::storage::{CompactReader, CompactWriter};

/// Batch ingestion helper — buffers vectors and flushes in chunks to amortize `write_all` syscalls.
pub struct BatchWriter {
    batch: Vec<Vec<f32>>,
    batch_bytes: usize,
    flush_threshold: usize,
}

impl BatchWriter {
    pub fn new(flush_threshold: usize) -> Self {
        Self {
            batch: Vec::new(),
            batch_bytes: 0,
            flush_threshold: flush_threshold.max(1),
        }
    }

    pub fn with_capacity(cap: usize, threshold: usize) -> Self {
        Self {
            batch: Vec::with_capacity(cap),
            batch_bytes: 0,
            flush_threshold: threshold.max(1),
        }
    }

    pub fn push(&mut self, vec: Vec<f32>) {
        self.batch_bytes += vec.len() * 4;
        self.batch.push(vec);
    }

    pub fn len(&self) -> usize {
        self.batch.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    pub fn should_flush(&self) -> bool {
        self.batch.len() >= self.flush_threshold
    }

    /// Flush buffered vectors to the writer; clears buffer.
    pub fn flush_to(&mut self, writer: &mut CompactWriter) -> Result<()> {
        for v in self.batch.drain(..) {
            writer.append(&v)?;
        }
        self.batch_bytes = 0;
        Ok(())
    }

    /// Flush remaining (call before `finalize`).
    pub fn finish(&mut self, writer: &mut CompactWriter) -> Result<()> {
        if !self.is_empty() {
            self.flush_to(writer)?;
        }
        Ok(())
    }
}

/// Parallel calibration using `std::thread::scope`.
/// Splits dataset into chunks, computes per-chunk min/max in parallel, then merges.
pub fn parallel_calibrate(vectors: &[Vec<f32>], num_threads: usize) -> Result<Quantizer> {
    if vectors.is_empty() {
        return Err(CompactError::EmptyDataset);
    }
    let dims = vectors[0].len();
    let threads = if num_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        num_threads
    }
    .min(vectors.len())
    .max(1);

    let chunk = (vectors.len() + threads - 1) / threads;

    let mut global_min = vec![f32::INFINITY; dims];
    let mut global_max = vec![f32::NEG_INFINITY; dims];

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for t in 0..threads {
            let start = t * chunk;
            let end = (start + chunk).min(vectors.len());
            if start >= end {
                continue;
            }
            let slice = &vectors[start..end];
            handles.push(s.spawn(move || {
                let mut mins = vec![f32::INFINITY; dims];
                let mut maxs = vec![f32::NEG_INFINITY; dims];
                for v in slice {
                    for (d, &val) in v.iter().enumerate() {
                        if val < mins[d] {
                            mins[d] = val;
                        }
                        if val > maxs[d] {
                            maxs[d] = val;
                        }
                    }
                }
                (mins, maxs)
            }));
        }
        for h in handles {
            if let Ok((mins, maxs)) = h.join() {
                for d in 0..dims {
                    if mins[d] < global_min[d] {
                        global_min[d] = mins[d];
                    }
                    if maxs[d] > global_max[d] {
                        global_max[d] = maxs[d];
                    }
                }
            }
        }
    });

    Quantizer::new(global_min, global_max)
}

/// Chunked reader: yields batches of dequantized vectors without allocating per full scan.
/// Reuses buffers and yields `Vec<Vec<f32>>` batches of `batch_size`.
pub struct ChunkedReader<'a> {
    reader: &'a CompactReader,
    next: u64,
    batch_size: usize,
    q_buf: Vec<u8>,
    f_buf: Vec<f32>,
}

impl<'a> ChunkedReader<'a> {
    pub fn new(reader: &'a CompactReader, batch_size: usize) -> Self {
        let dims = reader.dims();
        Self {
            reader,
            next: 0,
            batch_size: batch_size.max(1),
            q_buf: vec![0u8; dims],
            f_buf: vec![0f32; dims],
        }
    }
}

impl<'a> Iterator for ChunkedReader<'a> {
    type Item = Result<Vec<Vec<f32>>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.reader.len() {
            return None;
        }
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if self.next >= self.reader.len() {
                break;
            }
            if let Err(e) = self.reader.get_quantized_into(self.next, &mut self.q_buf) {
                return Some(Err(e));
            }
            if let Err(e) = self
                .reader
                .quantizer()
                .dequantize_into(&self.q_buf, &mut self.f_buf)
            {
                return Some(Err(e));
            }
            batch.push(self.f_buf.clone());
            self.next += 1;
        }
        Some(Ok(batch))
    }
}

/// Parallel batch search: split queries across threads, each thread does brute-force.
pub fn parallel_batch_search(
    reader: &CompactReader,
    queries: &[Vec<f32>],
    k: usize,
    num_threads: usize,
) -> Result<Vec<Vec<crate::search::SearchResult>>> {
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let threads = if num_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        num_threads
    }
    .min(queries.len())
    .max(1);

    let chunk = (queries.len() + threads - 1) / threads;
    let mut out: Vec<Vec<crate::search::SearchResult>> = Vec::with_capacity(queries.len());
    // We collect per thread then flatten in order.
    let mut chunks: Vec<Vec<Vec<crate::search::SearchResult>>> = Vec::new();

    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let start = t * chunk;
            let end = (start + chunk).min(queries.len());
            if start >= end {
                continue;
            }
            let slice = &queries[start..end];
            let r = reader;
            handles.push(s.spawn(move || {
                let mut local = Vec::with_capacity(slice.len());
                for q in slice {
                    let res = r.search(q, k).unwrap_or_default();
                    local.push(res);
                }
                local
            }));
        }
        for h in handles {
            if let Ok(v) = h.join() {
                chunks.push(v);
            }
        }
    });

    for c in chunks {
        out.extend(c);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{DistanceMetric, QuantType, Quantizer};

    #[test]
    fn batch_writer_flush() {
        let data = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.5, 0.5]];
        let q = Quantizer::calibrate(&data).expect("cal");
        let mut p = std::env::temp_dir();
        p.push(format!("batch_test_{}.btcp", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let mut w =
            CompactWriter::create(&p, q, QuantType::SQ8, DistanceMetric::L2).expect("create");
        let mut bw = BatchWriter::new(2);
        for v in data.clone() {
            bw.push(v);
            if bw.should_flush() {
                bw.flush_to(&mut w).expect("flush");
            }
        }
        bw.finish(&mut w).expect("finish");
        w.finalize().expect("fin");
        let r = CompactReader::open(&p).expect("open");
        assert_eq!(r.len(), 3);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn parallel_calibrate_matches() {
        let data: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32, i as f32 * 2.0]).collect();
        let q1 = Quantizer::calibrate(&data).expect("cal");
        let q2 = parallel_calibrate(&data, 4).expect("par");
        assert_eq!(q1.min_bounds(), q2.min_bounds());
        assert_eq!(q1.max_bounds(), q2.max_bounds());
    }

    #[test]
    fn chunked_reader() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let q = Quantizer::calibrate(&data).expect("cal");
        let mut p = std::env::temp_dir();
        p.push(format!("chunk_test_{}.btcp", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let mut w =
            CompactWriter::create(&p, q, QuantType::SQ8, DistanceMetric::L2).expect("create");
        for v in &data {
            w.append(v).expect("append");
        }
        w.finalize().expect("fin");
        let r = CompactReader::open(&p).expect("open");
        let mut iter = ChunkedReader::new(&r, 2);
        let b1 = iter.next().unwrap().expect("b1");
        assert_eq!(b1.len(), 2);
        let b2 = iter.next().unwrap().expect("b2");
        assert_eq!(b2.len(), 2);
        assert!(iter.next().is_none());
        let _ = std::fs::remove_file(&p);
    }
}
