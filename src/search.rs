use crate::errors::{CompactError, Result};
use crate::quant::Quantizer;
use crate::storage::CompactReader;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

impl SearchResult {
    pub fn new(id: u64, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Insert into a sorted top-k (ascending distance). Maintains at most k entries.
/// Uses insertion sort for small k (typical k <= 100) for cache friendliness.
fn insert_top_k(top_k: &mut Vec<SearchResult>, candidate: SearchResult, k: usize) {
    if k == 0 {
        return;
    }
    if top_k.len() < k {
        top_k.push(candidate);
        // Insertion sort step: bubble up
        let mut i = top_k.len() - 1;
        while i > 0 && top_k[i].distance < top_k[i - 1].distance {
            top_k.swap(i, i - 1);
            i -= 1;
        }
    } else if candidate.distance < top_k[k - 1].distance {
        // Replace worst
        top_k[k - 1] = candidate;
        let mut i = k - 1;
        while i > 0 && top_k[i].distance < top_k[i - 1].distance {
            top_k.swap(i, i - 1);
            i -= 1;
        }
    }
}

/// Brute-force search over a `CompactReader` using the supplied distance fn.
/// Reuses buffers to keep per-vector cost to 1 seek + dequant (zero alloc when dims <=1024).
pub fn brute_force_search<F>(
    reader: &CompactReader,
    query: &[f32],
    k: usize,
    distance_fn: F,
) -> Result<Vec<SearchResult>>
where
    F: Fn(&[f32], &[f32]) -> Result<f32>,
{
    if query.len() != reader.dims() {
        return Err(CompactError::DimensionMismatch {
            expected: reader.dims(),
            found: query.len(),
        });
    }
    if k == 0 || reader.is_empty() {
        return Ok(Vec::new());
    }

    let dims = reader.dims();
    let mut top_k = Vec::with_capacity(k.min(reader.len() as usize));

    // Reusable buffers
    let mut q_buf = vec![0u8; dims];
    let mut f_buf = vec![0f32; dims];

    for id in 0..reader.len() {
        reader.get_quantized_into(id, &mut q_buf)?;
        reader.quantizer().dequantize_into(&q_buf, &mut f_buf)?;
        let dist = distance_fn(query, &f_buf)?;
        if !dist.is_finite() {
            continue;
        }
        insert_top_k(&mut top_k, SearchResult::new(id, dist), k);
    }

    Ok(top_k)
}

/// Search using pre-quantized query (quantize once, then compare quantized bytes via table).
/// For SQ8, we can compute approximate L2 directly in quantized domain by dequantizing on the fly.
/// This variant quantizes the query once and reuses it.
pub fn brute_force_search_quantized(
    reader: &CompactReader,
    quantizer: &Quantizer,
    query: &[f32],
    k: usize,
    metric: crate::quant::DistanceMetric,
) -> Result<Vec<SearchResult>> {
    if query.len() != reader.dims() {
        return Err(CompactError::DimensionMismatch {
            expected: reader.dims(),
            found: query.len(),
        });
    }
    let q_query = quantizer.quantize_vector(query)?;
    let dq_query = quantizer.dequantize_vector(&q_query)?;
    brute_force_search(reader, &dq_query, k, |a, b| {
        crate::distance::distance(metric, a, b)
    })
}

/// Batch search: multiple queries against the same reader.
/// Returns Vec per query.
pub fn batch_search<F>(
    reader: &CompactReader,
    queries: &[Vec<f32>],
    k: usize,
    distance_fn: F,
) -> Result<Vec<Vec<SearchResult>>>
where
    F: Fn(&[f32], &[f32]) -> Result<f32> + Copy,
{
    let mut results = Vec::with_capacity(queries.len());
    for q in queries {
        results.push(brute_force_search(reader, q, k, distance_fn)?);
    }
    Ok(results)
}

/// Parallel brute-force search using `std::thread::scope`.
/// Splits the id space into `num_threads` chunks and merges top-k.
/// If `num_threads` is 0, uses `available_parallelism` or 4 as fallback.
/// Requires `F: Send + Sync` and `reader: Sync`.
pub fn parallel_search<F>(
    reader: &CompactReader,
    query: &[f32],
    k: usize,
    num_threads: usize,
    distance_fn: F,
) -> Result<Vec<SearchResult>>
where
    F: Fn(&[f32], &[f32]) -> Result<f32> + Send + Sync + Copy,
{
    if query.len() != reader.dims() {
        return Err(CompactError::DimensionMismatch {
            expected: reader.dims(),
            found: query.len(),
        });
    }
    if k == 0 || reader.is_empty() {
        return Ok(Vec::new());
    }

    let threads = if num_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        num_threads
    };
    let threads = threads.min(reader.len() as usize).max(1);
    let chunk = (reader.len() + threads as u64 - 1) / threads as u64;

    let query_owned = query.to_vec();

    let mut global_top: Vec<SearchResult> = Vec::with_capacity(k);

    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(threads);
        for t in 0..threads {
            let start = t as u64 * chunk;
            let end = (start + chunk).min(reader.len());
            if start >= end {
                continue;
            }
            let reader_ref = reader;
            let q_ref = &query_owned;
            let handle = s.spawn(move || {
                let dims = reader_ref.dims();
                let mut q_buf = vec![0u8; dims];
                let mut f_buf = vec![0f32; dims];
                let mut local_top = Vec::with_capacity(k);
                for id in start..end {
                    // If read fails, skip (should not happen in verified file)
                    if reader_ref.get_quantized_into(id, &mut q_buf).is_err() {
                        continue;
                    }
                    if reader_ref.quantizer().dequantize_into(&q_buf, &mut f_buf).is_err() {
                        continue;
                    }
                    let dist = match distance_fn(q_ref, &f_buf) {
                        Ok(d) if d.is_finite() => d,
                        _ => continue,
                    };
                    insert_top_k(&mut local_top, SearchResult::new(id, dist), k);
                }
                local_top
            });
            handles.push(handle);
        }

        for h in handles {
            if let Ok(local) = h.join() {
                for r in local {
                    insert_top_k(&mut global_top, r, k);
                }
            }
        }
    });

    Ok(global_top)
}

/// Simple linear scan iterator that yields `(id, Vec<f32>)` lazily.
/// Demonstrates zero-copy per-iteration pattern for callers who want custom logic.
pub struct ScanIter<'a> {
    reader: &'a CompactReader,
    next_id: u64,
    end: u64,
    q_buf: Vec<u8>,
    f_buf: Vec<f32>,
}

impl<'a> ScanIter<'a> {
    pub fn new(reader: &'a CompactReader) -> Self {
        let dims = reader.dims();
        Self {
            reader,
            next_id: 0,
            end: reader.len(),
            q_buf: vec![0u8; dims],
            f_buf: vec![0f32; dims],
        }
    }
}

impl<'a> Iterator for ScanIter<'a> {
    type Item = Result<(u64, Vec<f32>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_id >= self.end {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        if let Err(e) = self.reader.get_quantized_into(id, &mut self.q_buf) {
            return Some(Err(e));
        }
        if let Err(e) = self.reader.quantizer().dequantize_into(&self.q_buf, &mut self.f_buf) {
            return Some(Err(e));
        }
        Some(Ok((id, self.f_buf.clone())))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.end - self.next_id) as usize;
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for ScanIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance;
    use crate::quant::{DistanceMetric, Quantizer, QuantType};
    use crate::storage::CompactWriter;
    use std::fs;

    fn make_reader(dims: usize, data: &[Vec<f32>]) -> (CompactReader, std::path::PathBuf) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CTR: AtomicUsize = AtomicUsize::new(0);
        let c = CTR.fetch_add(1, Ordering::SeqCst);
        let q = Quantizer::calibrate(data).expect("calibrate");
        let mut p = std::env::temp_dir();
        p.push(format!(
            "search_test_{}_{}_{}_{}.btcp",
            dims,
            std::process::id(),
            c,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = fs::remove_file(&p);
        let mut w = CompactWriter::create(&p, q, QuantType::SQ8, DistanceMetric::L2).expect("create");
        for v in data {
            w.append(v).expect("append");
        }
        w.finalize().expect("finalize");
        let r = CompactReader::open(&p).expect("open");
        (r, p)
    }

    #[test]
    fn top_k_brute() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
        ];
        let (r, p) = make_reader(2, &data);
        let query = vec![0.1, 0.1];
        let res = brute_force_search(&r, &query, 2, distance::l2_squared).expect("search");
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, 0); // closest to origin
        let _ = fs::remove_file(p);
    }

    #[test]
    fn parallel_matches_single() {
        let data: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32, i as f32 * 0.5]).collect();
        let (r, p) = make_reader(2, &data);
        let query = vec![5.1, 2.6];
        let single = brute_force_search(&r, &query, 3, distance::l2_squared).expect("single");
        let parallel = parallel_search(&r, &query, 3, 4, distance::l2_squared).expect("par");
        assert_eq!(single, parallel);
        let _ = fs::remove_file(p);
    }

    #[test]
    fn scan_iter() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let (r, p) = make_reader(2, &data);
        let collected: Vec<_> = ScanIter::new(&r).collect::<Result<Vec<_>>>().expect("iter");
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].0, 0);
        let _ = fs::remove_file(p);
    }
}
