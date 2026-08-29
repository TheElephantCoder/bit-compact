use bit_compact::{distance, CompactReader, CompactWriter, DistanceMetric, QuantType, Quantizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Bigger synthetic dataset: 100 vectors, 8 dims in [-1,1]
    let dims = 8;
    let count = 100;
    let dataset: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            (0..dims)
                .map(|d| ((i * 7 + d * 13) % 100) as f32 / 50.0 - 1.0)
                .collect()
        })
        .collect();

    let q = Quantizer::calibrate(&dataset)?;
    let path = "/tmp/bitcompact_search.btcp";
    let _ = std::fs::remove_file(path);

    let mut w = CompactWriter::create(path, q.clone(), QuantType::SQ8, DistanceMetric::L2)?;
    for v in &dataset {
        w.append(v)?;
    }
    w.finalize()?;
    println!("wrote {count}x{dims} to {path}");

    let r = CompactReader::open(path)?;
    let query = vec![0.2, -0.1, 0.5, 0.0, 0.3, -0.4, 0.1, 0.9];

    // Single-thread brute force
    let top3 = bit_compact::brute_force_search(&r, &query, 3, distance::l2_squared)?;
    println!("top3 (single):");
    for hit in &top3 {
        println!("  id={} dist={:.4}", hit.id, hit.distance);
    }

    // Parallel
    let top3p = bit_compact::parallel_search(&r, &query, 3, 4, distance::l2_squared)?;
    println!("top3 (parallel 4 threads):");
    for hit in &top3p {
        println!("  id={} dist={:.4}", hit.id, hit.distance);
    }

    // Via reader convenience
    let top_via_reader = r.search(&query, 3)?;
    assert_eq!(top3, top_via_reader);
    println!("reader.search matches brute_force_search");

    // Cosine example
    let q2 = vec![1.0; dims];
    let cos = r.search(&q2, 2)?;
    println!(
        "cos-like search top2 for [1;8]: {:?}",
        cos.iter().map(|r| r.id).collect::<Vec<_>>()
    );

    // Batch search
    let queries = vec![query.clone(), vec![0.0; dims]];
    let batch = bit_compact::search::batch_search(&r, &queries, 2, distance::l2_squared)?;
    println!("batch search 2 queries -> {} results each", batch[0].len());

    // Iterator demo
    let mut count_iter = 0;
    for item in r.iter() {
        let (id, vec) = item?;
        let _ = (id, vec);
        count_iter += 1;
        if count_iter >= 2 {
            break;
        }
    }
    println!("iter first 2 ok, total len {}", r.len());

    Ok(())
}
