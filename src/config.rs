use crate::errors::{CompactError, Result};
use crate::quant::{DistanceMetric, QuantType};

/// Tunables for both writer and reader — `src/config.rs:5`
/// Mirrors the on-disk header plus runtime options (alignment, verification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactConfig {
    pub dims: usize,
    pub quant_type: QuantType,
    pub distance: DistanceMetric,
    pub major: u16,
    pub minor: u16,
    pub align_disk_blocks: bool,
    pub verify_on_open: bool,
}

impl CompactConfig {
    pub fn new(dims: usize, quant_type: QuantType, distance: DistanceMetric) -> Result<Self> {
        Self::builder(dims, quant_type, distance).build()
    }

    pub fn builder(dims: usize, quant_type: QuantType, distance: DistanceMetric) -> ConfigBuilder {
        ConfigBuilder {
            dims,
            quant_type,
            distance,
            major: crate::VERSION_MAJOR,
            minor: crate::VERSION_MINOR,
            align_disk_blocks: false,
            verify_on_open: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.dims == 0 {
            return Err(CompactError::invalid_header("dims must be > 0"));
        }
        if self.dims > u16::MAX as usize {
            return Err(CompactError::invalid_header(format!(
                "dims {} exceeds u16::MAX",
                self.dims
            )));
        }
        Ok(())
    }

    #[inline]
    pub fn header_dims_u16(&self) -> Result<u16> {
        self.validate()?;
        Ok(self.dims as u16)
    }
}

/// Builder for `CompactConfig` — fluent, fallible only on `build()`.
#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    dims: usize,
    quant_type: QuantType,
    distance: DistanceMetric,
    major: u16,
    minor: u16,
    align_disk_blocks: bool,
    verify_on_open: bool,
}

impl ConfigBuilder {
    pub fn version(mut self, major: u16, minor: u16) -> Self {
        self.major = major;
        self.minor = minor;
        self
    }

    pub fn align_disk_blocks(mut self, align: bool) -> Self {
        self.align_disk_blocks = align;
        self
    }

    pub fn verify_on_open(mut self, verify: bool) -> Self {
        self.verify_on_open = verify;
        self
    }

    pub fn quant_type(mut self, q: QuantType) -> Self {
        self.quant_type = q;
        self
    }

    pub fn distance(mut self, d: DistanceMetric) -> Self {
        self.distance = d;
        self
    }

    pub fn build(self) -> Result<CompactConfig> {
        let cfg = CompactConfig {
            dims: self.dims,
            quant_type: self.quant_type,
            distance: self.distance,
            major: self.major,
            minor: self.minor,
            align_disk_blocks: self.align_disk_blocks,
            verify_on_open: self.verify_on_open,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Writer-specific knobs that extend `CompactConfig`.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    pub base: CompactConfig,
    /// If true, `finalize()` pads to next 4096B boundary before footer.
    pub pad_to_block: bool,
    /// Optional pre-allocated buffer size for batched appends.
    pub batch_capacity: usize,
}

impl WriterConfig {
    pub fn from_base(base: CompactConfig) -> Self {
        let pad = base.align_disk_blocks;
        Self {
            base,
            pad_to_block: pad,
            batch_capacity: 1024,
        }
    }

    pub fn pad_to_block(mut self, pad: bool) -> Self {
        self.pad_to_block = pad;
        self
    }

    pub fn batch_capacity(mut self, cap: usize) -> Self {
        self.batch_capacity = cap.max(1);
        self
    }
}

/// Reader-specific knobs.
#[derive(Debug, Clone, Copy)]
pub struct ReaderConfig {
    pub verify_checksum: bool,
    pub prefetch_blocks: usize,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        Self {
            verify_checksum: true,
            prefetch_blocks: 0,
        }
    }
}

impl ReaderConfig {
    pub fn without_verification(mut self) -> Self {
        self.verify_checksum = false;
        self
    }

    pub fn with_prefetch(mut self, blocks: usize) -> Self {
        self.prefetch_blocks = blocks;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_ok() {
        let cfg = CompactConfig::builder(128, QuantType::SQ8, DistanceMetric::L2)
            .version(1, 0)
            .align_disk_blocks(true)
            .build()
            .expect("build");
        assert_eq!(cfg.dims, 128);
        assert!(cfg.align_disk_blocks);
    }

    #[test]
    fn zero_dims_rejected() {
        let err = CompactConfig::builder(0, QuantType::SQ8, DistanceMetric::L2)
            .build()
            .expect_err("should fail");
        assert!(format!("{err}").contains("dims"));
    }

    #[test]
    fn exceeds_u16_rejected() {
        let err = CompactConfig::builder(70000, QuantType::SQ8, DistanceMetric::L2)
            .build()
            .expect_err("should fail");
        assert!(format!("{err}").contains("u16"));
    }
}
