use crate::errors::{CompactError, Result};
use crate::storage::CompactReader;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Export helpers — `src/export.rs:3`
pub fn export_json(reader: &CompactReader, path: &Path) -> Result<()> {
    let mut file = File::create(path).map_err(|e| CompactError::IoError { source: e })?;
    writeln!(file, "[").map_err(|e| CompactError::IoError { source: e })?;
    for i in 0..reader.len() {
        let v = reader.get_vector(i)?;
        let line = format!("  [{}]{}", v.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>().join(", "), if i+1==reader.len() {""} else {","});
        writeln!(file, "{}", line).map_err(|e| CompactError::IoError { source: e })?;
    }
    writeln!(file, "]").map_err(|e| CompactError::IoError { source: e })?;
    Ok(())
}

pub fn export_csv(reader: &CompactReader, path: &Path) -> Result<()> {
    let mut file = File::create(path).map_err(|e| CompactError::IoError { source: e })?;
    for i in 0..reader.len() {
        let v = reader.get_vector(i)?;
        let line = v.iter().map(|x| format!("{:.6}", x)).collect::<Vec<_>>().join(",");
        writeln!(file, "{}", line).map_err(|e| CompactError::IoError { source: e })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{Quantizer, QuantType, DistanceMetric};
    use crate::storage::CompactWriter;
    use std::fs;
    #[test]
    fn export_roundtrip() {
        let data = vec![vec![0.0,1.0], vec![1.0,0.0]];
        let q = Quantizer::calibrate(&data).unwrap();
        let mut p = std::env::temp_dir(); p.push(format!("export_{}.btcp", std::process::id()));
        let _ = fs::remove_file(&p);
        let mut w=CompactWriter::create(&p,q,QuantType::SQ8,DistanceMetric::L2).unwrap();
        for v in &data { w.append(v).unwrap(); }
        w.finalize().unwrap();
        let r=CompactReader::open(&p).unwrap();
        let mut out=std::env::temp_dir(); out.push(format!("out_{}.json", std::process::id()));
        export_json(&r,&out).unwrap();
        assert!(out.exists());
        let _=fs::remove_file(&p); let _=fs::remove_file(out);
    }
}
