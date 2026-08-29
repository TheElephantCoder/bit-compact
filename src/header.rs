use crate::errors::{CompactError, Result};
use crate::quant::{DistanceMetric, QuantType};

pub const MAGIC: [u8; 4] = *b"BTCP";
pub const HEADER_SIZE: usize = 32;
pub const DISK_BLOCK_SIZE: usize = 4096;

/// On-disk header — `src/header.rs:8` — exactly 32B BE.
/// Layout: magic 4 | major 2 | minor 2 | dims 2 | quant 2 | distance 2 | reserved 2 | count 8 | footer 8
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub major: u16,
    pub minor: u16,
    pub dims: u16,
    pub quant_type: QuantType,
    pub distance_metric: DistanceMetric,
    pub vector_count: u64,
    pub footer_offset: u64,
}

impl Header {
    pub fn new(
        major: u16,
        minor: u16,
        dims: u16,
        quant_type: QuantType,
        distance_metric: DistanceMetric,
        vector_count: u64,
        footer_offset: u64,
    ) -> Self {
        Self {
            major,
            minor,
            dims,
            quant_type,
            distance_metric,
            vector_count,
            footer_offset,
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4..6].copy_from_slice(&self.major.to_be_bytes());
        buf[6..8].copy_from_slice(&self.minor.to_be_bytes());
        buf[8..10].copy_from_slice(&self.dims.to_be_bytes());
        buf[10..12].copy_from_slice(&self.quant_type.as_u16().to_be_bytes());
        buf[12..14].copy_from_slice(&self.distance_metric.as_u16().to_be_bytes());
        // 14..16 reserved stays 0
        buf[16..24].copy_from_slice(&self.vector_count.to_be_bytes());
        buf[24..32].copy_from_slice(&self.footer_offset.to_be_bytes());
        buf
    }

    pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Result<Self> {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if magic != MAGIC {
            return Err(CompactError::InvalidMagicBytes {
                expected: MAGIC,
                found: magic,
            });
        }
        let major = u16::from_be_bytes([bytes[4], bytes[5]]);
        let minor = u16::from_be_bytes([bytes[6], bytes[7]]);
        let dims = u16::from_be_bytes([bytes[8], bytes[9]]);
        if dims == 0 {
            return Err(CompactError::invalid_header("dims must be > 0"));
        }
        let quant_raw = u16::from_be_bytes([bytes[10], bytes[11]]);
        let dist_raw = u16::from_be_bytes([bytes[12], bytes[13]]);
        let vector_count = u64::from_be_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);
        let footer_offset = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let quant_type = QuantType::from_u16(quant_raw)?;
        let distance_metric = DistanceMetric::from_u16(dist_raw)?;
        Ok(Self {
            major,
            minor,
            dims,
            quant_type,
            distance_metric,
            vector_count,
            footer_offset,
        })
    }

    #[inline]
    pub fn metadata_len(&self) -> usize {
        self.dims as usize * 8
    }

    #[inline]
    pub fn data_len(&self) -> u64 {
        self.vector_count * self.dims as u64
    }

    pub fn validate_footer(&self, file_len: u64) -> Result<()> {
        let min_footer = HEADER_SIZE as u64 + self.metadata_len() as u64 + self.data_len();
        if self.footer_offset < min_footer {
            return Err(CompactError::CorruptedFooter(format!(
                "footer {} < min {}",
                self.footer_offset, min_footer
            )));
        }
        // footer = row_ids (count*8) + checksum 32
        let footer_size = self.vector_count * 8 + 32;
        if self.footer_offset + footer_size > file_len {
            return Err(CompactError::CorruptedFooter(format!(
                "footer extends beyond file: offset {} + size {} > file {}",
                self.footer_offset, footer_size, file_len
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let h = Header::new(1, 2, 128, QuantType::SQ8, DistanceMetric::Cosine, 10, 5000);
        let b = h.to_bytes();
        assert_eq!(b.len(), 32);
        let p = Header::from_bytes(&b).expect("parse");
        assert_eq!(h, p);
    }

    #[test]
    fn bad_magic() {
        let mut b = Header::new(1, 0, 8, QuantType::SQ8, DistanceMetric::L2, 0, 0).to_bytes();
        b[0] = 0;
        assert!(Header::from_bytes(&b).is_err());
    }
}
