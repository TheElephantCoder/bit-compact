use bit_compact::{distance, Quantizer};

fn main() {
    // Cheap bench without criterion (zero deps). Run with `cargo run --release --bench quant_bench`
    let dims = 128;
    let n = 10_000;
    let data: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dims)
                .map(|d| ((i + d) % 100) as f32 / 50.0 - 1.0)
                .collect()
        })
        .collect();
    let q = Quantizer::calibrate(&data).expect("calibrate");

    let start = std::time::Instant::now();
    let mut sink = 0u64;
    for v in &data {
        let qb = q.quantize_vector(v).expect("q");
        sink += qb.len() as u64;
    }
    let elapsed = start.elapsed();
    println!(
        "quantize {n}x{dims}: {:?} ({:.0} vec/s) sink {sink}",
        elapsed,
        n as f64 / elapsed.as_secs_f64()
    );

    let start = std::time::Instant::now();
    let mut sum = 0.0;
    for i in 0..n - 1 {
        sum += distance::l2_squared(&data[i], &data[i + 1]).unwrap_or(0.0);
    }
    println!("l2_squared {n} pairs: {:?} sum {sum:.2}", start.elapsed());
}
