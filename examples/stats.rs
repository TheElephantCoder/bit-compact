use bit_compact::{stats, Quantizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![0.5, 0.5, 0.5],
        vec![-1.0, 2.0, 0.0],
        vec![0.2, 0.8, -0.5],
    ];

    let q = Quantizer::calibrate(&data)?;
    let report = stats::evaluate(&q, &data);
    println!("{}", report.summary());
    println!("per-dim mse: {:?}", report.per_dim_mse);
    println!("per-dim range: {:?}", report.per_dim_range);
    println!(
        "theoretical max error dim0 range {} -> {:.6}",
        report.per_dim_range[0],
        stats::theoretical_max_error(report.per_dim_range[0])
    );
    println!(
        "compression {}B -> {}B = {:.2}x",
        report.original_bytes, report.compressed_bytes, report.compression_ratio
    );

    // Global vs per-dim vs percentile
    let q_global = Quantizer::calibrate_global(&data)?;
    let rep_global = stats::evaluate(&q_global, &data);
    println!(
        "\nglobal calibrate mse={:.6} vs per-dim mse={:.6}",
        rep_global.mse, report.mse
    );

    let q_pct = Quantizer::calibrate_percentile(&data, 5.0, 95.0)?;
    let rep_pct = stats::evaluate(&q_pct, &data);
    println!(
        "percentile 5-95 mse={:.6} snr={:.2}dB",
        rep_pct.mse, rep_pct.snr_db
    );

    // Quantize batch
    let batch_q = q.quantize_batch(&data)?;
    let batch_dq = q.dequantize_batch(&batch_q)?;
    println!(
        "\nbatch roundtrip first vec orig {:?} dq {:?}",
        data[0], batch_dq[0]
    );

    // Aligned buffer demo
    let buf = bit_compact::CacheAlignedBuffer::new_zeroed(128);
    println!(
        "cache aligned buf len={} aligned={}",
        buf.len(),
        bit_compact::aligned::is_cache_aligned(buf.as_ptr())
    );

    Ok(())
}
