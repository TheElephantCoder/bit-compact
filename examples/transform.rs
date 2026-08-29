use bit_compact::{transform_dataset, Centering, Normalizer, Standardizer, Transform};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = vec![vec![1.0, 10.0], vec![3.0, 20.0], vec![5.0, 30.0]];

    // Center
    let cent = Centering::from_data(&data)?;
    println!("mean: {:?}", cent.mean());
    let centered = transform_dataset(&cent, &data)?;
    println!("centered: {:?}", centered);

    // Standardize
    let std = Standardizer::from_data(&data)?;
    println!("std mean {:?} std {:?}", std.mean(), std.std());
    let standardized = transform_dataset(&std, &data)?;
    println!("standardized first vec {:?}", standardized[0]);

    // Normalize to unit
    let norm = Normalizer::new(2);
    let mut out = vec![0.0; 2];
    norm.transform(&[3.0, 4.0], &mut out)?;
    println!(
        "normalize [3,4] -> {:?} norm {}",
        out,
        (out[0] * out[0] + out[1] * out[1]).sqrt()
    );

    // Chain: center then normalize
    let chain = bit_compact::transform::Chain::new(cent, norm)?;
    let mut out2 = vec![0.0; 2];
    chain.transform(&[1.0, 10.0], &mut out2)?;
    println!("chain center+norm [1,10] -> {:?}", out2);

    Ok(())
}
