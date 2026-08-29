use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::errors::{CompactError, Result};
use crate::header::{Header, HEADER_SIZE};
use crate::quant::Quantizer;

/// Validation report — `src/validate.rs:8`
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub path: String,
    pub header: Header,
    pub dims: usize,
    pub count: u64,
    pub footer_offset: u64,
    pub file_size: u64,
    pub checksum_valid: bool,
    pub row_ids_monotonic: bool,
    pub metadata_finite: bool,
    pub footer_size_ok: bool,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.checksum_valid
            && self.row_ids_monotonic
            && self.metadata_finite
            && self.footer_size_ok
            && self.warnings.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "file={} dims={} count={} footer={} size={} valid={} checksum={} rows_monotonic={} meta_finite={} warnings={:?}",
            self.path,
            self.dims,
            self.count,
            self.footer_offset,
            self.file_size,
            self.is_valid(),
            self.checksum_valid,
            self.row_ids_monotonic,
            self.metadata_finite,
            self.warnings
        )
    }
}

/// Validate a bit-compact file without fully loading data (except checksum).
/// Checks header magic, dims, footer bounds, row_ids monotonic, checksum, metadata finiteness.
pub fn validate<P: AsRef<Path>>(path: P) -> Result<ValidationReport> {
    let path_ref = path.as_ref();
    let mut file = File::open(path_ref).map_err(|e| CompactError::IoError { source: e })?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

    // Header
    let mut header_buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut header_buf)
        .map_err(|e| CompactError::IoError { source: e })?;
    let header = Header::from_bytes(&header_buf)?;
    let dims = header.dims as usize;
    let count = header.vector_count;
    let footer_offset = header.footer_offset;
    let metadata_len = dims * 8;

    let mut warnings = Vec::new();

    // Footer bounds
    let min_footer = HEADER_SIZE as u64 + metadata_len as u64 + count * dims as u64;
    let footer_size_ok = footer_offset >= min_footer && footer_offset + count * 8 + 32 <= file_size;
    if !footer_size_ok {
        warnings.push(format!(
            "footer offset {} not in [{}, {}]",
            footer_offset,
            min_footer,
            file_size.saturating_sub(count * 8 + 32)
        ));
    }

    // Metadata finite check
    let mut metadata_finite = true;
    file.seek(SeekFrom::Start(HEADER_SIZE as u64))
        .map_err(|e| CompactError::IoError { source: e })?;
    let mut min_bounds = Vec::with_capacity(dims);
    let mut max_bounds = Vec::with_capacity(dims);
    for _ in 0..dims {
        let mut b = [0u8; 4];
        if file.read_exact(&mut b).is_err() {
            metadata_finite = false;
            warnings.push("truncated min bounds".into());
            break;
        }
        let v = f32::from_be_bytes(b);
        if !v.is_finite() {
            metadata_finite = false;
            warnings.push(format!("non-finite min {v}"));
        }
        min_bounds.push(v);
    }
    for _ in 0..dims {
        let mut b = [0u8; 4];
        if file.read_exact(&mut b).is_err() {
            metadata_finite = false;
            warnings.push("truncated max bounds".into());
            break;
        }
        let v = f32::from_be_bytes(b);
        if !v.is_finite() {
            metadata_finite = false;
            warnings.push(format!("non-finite max {v}"));
        }
        max_bounds.push(v);
    }
    // also check Quantizer construction
    if Quantizer::new(min_bounds.clone(), max_bounds.clone()).is_err() {
        metadata_finite = false;
        warnings.push("quantizer bounds invalid (max < min or dims mismatch)".into());
    }

    // Row IDs monotonic + checksum
    let mut row_ids_monotonic = true;
    let mut checksum_valid = true;
    if footer_size_ok {
        file.seek(SeekFrom::Start(footer_offset))
            .map_err(|e| CompactError::IoError { source: e })?;
        let mut prev: Option<u64> = None;
        for i in 0..count {
            let mut b = [0u8; 8];
            if file.read_exact(&mut b).is_err() {
                row_ids_monotonic = false;
                warnings.push(format!("truncated row id at {i}"));
                break;
            }
            let id = u64::from_be_bytes(b);
            if let Some(p) = prev {
                if id <= p {
                    row_ids_monotonic = false;
                    warnings.push(format!(
                        "row ids not strictly increasing: {p} -> {id} at {i}"
                    ));
                }
            }
            if id != i {
                // spec says row_ids are 0..count; allow but warn if not
                warnings.push(format!("row id {id} != expected {i}"));
            }
            prev = Some(id);
        }
        // Checksum
        let mut stored = [0u8; 32];
        if file.read_exact(&mut stored).is_err() {
            checksum_valid = false;
            warnings.push("truncated checksum".into());
        } else {
            // Recompute data block hash
            let data_start = (HEADER_SIZE + metadata_len) as u64;
            let data_len = count * dims as u64;
            let mut hasher = crate::sha::Sha256::new();
            file.seek(SeekFrom::Start(data_start))
                .map_err(|e| CompactError::IoError { source: e })?;
            const CHUNK: usize = 64 * 1024;
            let mut buf = vec![0u8; CHUNK];
            let mut remaining = data_len;
            let mut computed_ok = true;
            while remaining > 0 {
                let to_read = (remaining as usize).min(CHUNK);
                if buf.len() != to_read {
                    buf.resize(to_read, 0);
                }
                if file.read_exact(&mut buf).is_err() {
                    computed_ok = false;
                    checksum_valid = false;
                    warnings.push("truncated data block during checksum".into());
                    break;
                }
                hasher.update(&buf);
                remaining -= to_read as u64;
            }
            if computed_ok {
                let computed = hasher.finalize();
                if computed != stored {
                    checksum_valid = false;
                    warnings.push(format!(
                        "checksum mismatch expected {:02x?}.. found {:02x?}..",
                        &stored[..4],
                        &computed[..4]
                    ));
                }
            }
        }
    } else {
        checksum_valid = false;
        row_ids_monotonic = false;
    }

    Ok(ValidationReport {
        path: path_ref.display().to_string(),
        header,
        dims,
        count,
        footer_offset,
        file_size,
        checksum_valid,
        row_ids_monotonic,
        metadata_finite,
        footer_size_ok,
        warnings,
    })
}

/// Quick check: is file likely valid (cheap, verifies header + footer bounds only, not checksum)?
pub fn quick_check<P: AsRef<Path>>(path: P) -> Result<bool> {
    let mut file = File::open(path.as_ref()).map_err(|e| CompactError::IoError { source: e })?;
    let mut buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut buf)
        .map_err(|e| CompactError::IoError { source: e })?;
    let header = Header::from_bytes(&buf)?;
    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    Ok(header.validate_footer(file_len).is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{DistanceMetric, QuantType, Quantizer};
    use crate::storage::CompactWriter;
    use std::fs;

    #[test]
    fn validate_ok() {
        let data = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let q = Quantizer::calibrate(&data).expect("cal");
        let mut p = std::env::temp_dir();
        p.push(format!("validate_ok_{}.btcp", std::process::id()));
        let _ = fs::remove_file(&p);
        let mut w =
            CompactWriter::create(&p, q, QuantType::SQ8, DistanceMetric::L2).expect("create");
        for v in &data {
            w.append(v).expect("append");
        }
        w.finalize().expect("fin");
        let rep = validate(&p).expect("validate");
        assert!(rep.is_valid(), "report not valid: {}", rep.summary());
        assert!(rep.checksum_valid);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn quick_check_bad() {
        let mut p = std::env::temp_dir();
        p.push(format!("validate_bad_{}.btcp", std::process::id()));
        let _ = fs::remove_file(&p);
        std::fs::write(&p, [0u8; 32]).expect("write");
        assert!(validate(&p).is_err() || !validate(&p).unwrap().is_valid());
        let _ = fs::remove_file(&p);
    }
}
