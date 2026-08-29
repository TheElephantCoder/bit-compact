use bit_compact::{CompactReader, CompactWriter, DistanceMetric, QuantType, Quantizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Synthetic 3-dim embeddings
    let dataset = vec![
        vec![0.0, 1.0, 2.0],
        vec![3.0, 4.0, 5.0],
        vec![-1.0, 0.5, 10.0],
        vec![2.2, 3.3, 4.4],
    ];

    // Calibrate per-dimension bounds
    let quantizer = Quantizer::calibrate(&dataset)?;
    println!(
        "dims={} range sample dim0 [{}, {}]",
        quantizer.dims(),
        quantizer.min_bounds()[0],
        quantizer.max_bounds()[0]
    );

    let path = "/tmp/bitcompact_basic.btcp";
    let _ = std::fs::remove_file(path);

    // Writer with default SQ8 + L2
    let mut w = CompactWriter::create(path, quantizer.clone(), QuantType::SQ8, DistanceMetric::L2)?;
    for v in &dataset {
        w.append(v)?;
    }
    w.finalize()?;
    println!("wrote {} vectors to {path}", dataset.len());

    // Reader — 1-seek random access
    let r = CompactReader::open(path)?;
    println!(
        "read header: dims={} count={} footer_offset={}",
        r.dims(),
        r.len(),
        r.footer_offset()
    );

    // Zero-alloc quantized read
    let mut qbuf = vec![0u8; r.dims()];
    r.get_quantized_into(1, &mut qbuf)?;
    println!("quantized[1]={:?}", qbuf);

    // Dequantized
    let deq = r.get_vector(1)?;
    println!(
        "dequant[1]={:?} orig {:?} err {:?}",
        deq,
        dataset[1],
        vec![
            deq[0] - dataset[1][0],
            deq[1] - dataset[1][1],
            deq[2] - dataset[1][2]
        ]
    );

    // Batch
    let batch = r.get_batch(&[0, 2])?;
    println!("batch [0,2]={:?}", batch);

    // Config builder alternative
    let cfg = bit_compact::CompactConfig::builder(3, QuantType::SQ8, DistanceMetric::Cosine)
        .align_disk_blocks(true)
        .build()?;
    println!("config dims={} align={}", cfg.dims, cfg.align_disk_blocks);

    Ok(())
}
