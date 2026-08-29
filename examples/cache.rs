use bit_compact::{CachedReader, CompactReader, CompactWriter, DistanceMetric, QuantType, Quantizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.5, 0.5], vec![0.9, 0.1]];
    let q = Quantizer::calibrate(&data)?;
    let path = "/tmp/bitcompact_cache.btcp";
    let _ = std::fs::remove_file(path);
    let mut w = CompactWriter::create(path, q, QuantType::SQ8, DistanceMetric::L2)?;
    for v in &data { w.append(v)?; }
    w.finalize()?;

    let r = CompactReader::open(path)?;
    let cr = CachedReader::new(r, 2); // cache 2 hot vectors

    // Repeated access hits cache
    for _ in 0..5 {
        let v = cr.get(0)?;
        assert_eq!(v.len(), 2);
    }
    println!("cache len {} hit_rate {:.2}", cr.cache_len(), cr.hit_rate());

    let v2 = cr.get(1)?;
    println!("get 1: {:?}", v2);
    let v3 = cr.get(2)?;
    println!("get 2 (evicts): {:?}", v3);
    println!("hit_rate after mixed: {:.2}", cr.hit_rate());

    // batch with cache
    let batch = cr.get_batch(&[0, 2, 1])?;
    println!("batch via cache: {:?}", batch);

    Ok(())
}
