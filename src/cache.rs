use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::errors::Result;
use crate::storage::CompactReader;

/// Simple LRU cache for hot vectors — `src/cache.rs:5`
/// Capacity is in number of vectors (each `dims * 4` bytes). Thread-safe via `Mutex`.
#[derive(Debug)]
pub struct LruCache {
    capacity: usize,
    map: HashMap<u64, Vec<f32>>,
    order: VecDeque<u64>, // front = LRU, back = MRU
}

impl LruCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn contains(&self, id: u64) -> bool {
        self.map.contains_key(&id)
    }

    pub fn get(&mut self, id: u64) -> Option<Vec<f32>> {
        if let Some(v) = self.map.get(&id).cloned() {
            // move to MRU
            self.promote(id);
            Some(v)
        } else {
            None
        }
    }

    pub fn put(&mut self, id: u64, vec: Vec<f32>) {
        if self.map.contains_key(&id) {
            self.map.insert(id, vec);
            self.promote(id);
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(lru) = self.order.pop_front() {
                self.map.remove(&lru);
            }
        }
        self.order.push_back(id);
        self.map.insert(id, vec);
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    fn promote(&mut self, id: u64) {
        if let Some(pos) = self.order.iter().position(|&x| x == id) {
            self.order.remove(pos);
            self.order.push_back(id);
        }
    }

    pub fn hit_rate(&self, hits: u64, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

/// Reader wrapper that caches hot vectors in an `LruCache`.
///
/// `CachedReader` is `Send + Sync` if the inner `CompactReader` is.
/// Cache is shared via `Arc<Mutex<_>>` so concurrent threads benefit.
#[derive(Clone)]
pub struct CachedReader {
    inner: Arc<CompactReader>,
    cache: Arc<Mutex<LruCache>>,
    hits: Arc<Mutex<u64>>,
    total: Arc<Mutex<u64>>,
}

impl CachedReader {
    pub fn new(reader: CompactReader, capacity: usize) -> Self {
        Self {
            inner: Arc::new(reader),
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            hits: Arc::new(Mutex::new(0)),
            total: Arc::new(Mutex::new(0)),
        }
    }

    pub fn from_arc(reader: Arc<CompactReader>, capacity: usize) -> Self {
        Self {
            inner: reader,
            cache: Arc::new(Mutex::new(LruCache::new(capacity))),
            hits: Arc::new(Mutex::new(0)),
            total: Arc::new(Mutex::new(0)),
        }
    }

    pub fn inner(&self) -> &CompactReader {
        &self.inner
    }

    pub fn cache_len(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    pub fn hit_rate(&self) -> f64 {
        let h = *self.hits.lock().unwrap_or_else(|e| e.into_inner());
        let t = *self.total.lock().unwrap_or_else(|e| e.into_inner());
        if t == 0 {
            0.0
        } else {
            h as f64 / t as f64
        }
    }

    pub fn clear_cache(&self) {
        if let Ok(mut c) = self.cache.lock() {
            c.clear();
        }
    }

    /// Get vector, using cache if present; otherwise read from disk (1 seek) and insert.
    pub fn get(&self, id: u64) -> Result<Vec<f32>> {
        // stats
        if let Ok(mut t) = self.total.lock() {
            *t += 1;
        }
        // check cache
        if let Ok(mut cache) = self.cache.lock() {
            if let Some(v) = cache.get(id) {
                if let Ok(mut h) = self.hits.lock() {
                    *h += 1;
                }
                return Ok(v);
            }
        }
        // miss: read
        let vec = self.inner.get_vector(id)?;
        if let Ok(mut cache) = self.cache.lock() {
            cache.put(id, vec.clone());
        }
        Ok(vec)
    }

    /// Batch get with per-item cache lookup.
    pub fn get_batch(&self, ids: &[u64]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            out.push(self.get(id)?);
        }
        Ok(out)
    }
}

impl std::fmt::Debug for CachedReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedReader")
            .field("inner", &self.inner)
            .field("cache_len", &self.cache_len())
            .field("hit_rate", &self.hit_rate())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{DistanceMetric, QuantType, Quantizer};
    use crate::storage::CompactWriter;
    use std::fs;

    fn make_reader() -> (CompactReader, std::path::PathBuf) {
        let data = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.5, 0.5]];
        let q = Quantizer::calibrate(&data).expect("cal");
        let mut p = std::env::temp_dir();
        p.push(format!("cache_test_{}.btcp", std::process::id()));
        let _ = fs::remove_file(&p);
        let mut w =
            CompactWriter::create(&p, q, QuantType::SQ8, DistanceMetric::L2).expect("create");
        for v in &data {
            w.append(v).expect("append");
        }
        w.finalize().expect("fin");
        let r = CompactReader::open(&p).expect("open");
        (r, p)
    }

    #[test]
    fn lru_basic() {
        let mut c = LruCache::new(2);
        c.put(0, vec![1.0]);
        c.put(1, vec![2.0]);
        assert_eq!(c.len(), 2);
        c.put(2, vec![3.0]); // evicts 0
        assert!(!c.contains(0));
        assert!(c.contains(1));
        assert!(c.contains(2));
        // access 1 promotes it, then 3 evicts 2
        let _ = c.get(1);
        c.put(3, vec![4.0]);
        assert!(!c.contains(2));
        assert!(c.contains(1));
    }

    #[test]
    fn cached_reader_hit() {
        let (r, p) = make_reader();
        let cr = CachedReader::new(r, 2);
        let v0 = cr.get(0).expect("get");
        assert_eq!(cr.cache_len(), 1);
        let v0b = cr.get(0).expect("get2");
        assert_eq!(v0, v0b);
        assert!(cr.hit_rate() > 0.0);
        let _ = fs::remove_file(p);
    }
}
