//! Production-grade error subsystem — `src/errors.rs:1`
//! All errors are zero-dependency, `std`-only. No `unwrap()` is used internally.

use std::error::Error as StdError;
use std::fmt;
use std::io;

/// Crate-wide `Result` alias — `src/errors.rs:10`
pub type Result<T> = std::result::Result<T, CompactError>;

/// Exhaustive error taxonomy for bit-compact — `src/errors.rs:14`
///
/// Variants required by the specification are present verbatim:
/// `IoError`, `InvalidMagicBytes`, `DimensionMismatch`,
/// `CorruptedFooter`, `QuantizationOverflow`.
/// Additional variants cover header validation, checksums, and bounds.
#[derive(Debug)]
pub enum CompactError {
    /// Underlying I/O failure — maps `std::io::Error` losslessly.
    IoError { source: io::Error },

    /// File does not begin with `BTCP` magic — `src/errors.rs:26`
    InvalidMagicBytes { expected: [u8; 4], found: [u8; 4] },

    /// Header field violates invariant (version, counts, offsets).
    InvalidHeader(String),

    /// Vector dimensionality does not match engine calibration.
    DimensionMismatch { expected: usize, found: usize },

    /// Footer / index block could not be parsed or is internally inconsistent.
    CorruptedFooter(String),

    /// Data block checksum does not match footer SHA-256.
    ChecksumMismatch { expected: [u8; 32], found: [u8; 32] },

    /// Attempted to access an out-of-bounds vector index.
    IndexOutOfBounds { index: u64, count: u64 },

    /// Quantization produced a value outside the `0..=255` clamp range
    /// or consumed a non-finite input (NaN / ±∞).
    QuantizationOverflow {
        dimension: usize,
        value: f32,
        min: f32,
        max: f32,
        reason: &'static str,
    },

    /// Unknown quantization type disk tag.
    InvalidQuantType(u16),

    /// Unknown distance metric disk tag.
    InvalidDistanceMetric(u16),

    /// No vectors supplied to a calibrating or writing path.
    EmptyDataset,

    /// Generic corruption of the dense data block.
    CorruptedData(String),
}

impl fmt::Display for CompactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IoError { source } => write!(f, "I/O error: {source}"),
            Self::InvalidMagicBytes { expected, found } => write!(
                f,
                "invalid magic bytes: expected {:?} (\"{}\"), found {:?} (\"{}\")",
                expected,
                String::from_utf8_lossy(expected),
                found,
                String::from_utf8_lossy(found)
            ),
            Self::InvalidHeader(msg) => write!(f, "invalid header: {msg}"),
            Self::DimensionMismatch { expected, found } => write!(
                f,
                "dimension mismatch: expected {expected}, found {found}"
            ),
            Self::CorruptedFooter(msg) => write!(f, "corrupted footer: {msg}"),
            Self::ChecksumMismatch { expected, found } => write!(
                f,
                "checksum mismatch: expected {:02x?}, found {:02x?}",
                &expected[..8],
                &found[..8]
            ),
            Self::IndexOutOfBounds { index, count } => {
                write!(f, "index out of bounds: index {index} >= count {count}")
            }
            Self::QuantizationOverflow {
                dimension,
                value,
                min,
                max,
                reason,
            } => write!(
                f,
                "quantization overflow at dim {dimension}: value {value} not in [{min}, {max}] ({reason})"
            ),
            Self::InvalidQuantType(tag) => write!(f, "invalid quantization type tag: {tag}"),
            Self::InvalidDistanceMetric(tag) => write!(f, "invalid distance metric tag: {tag}"),
            Self::EmptyDataset => write!(f, "empty dataset: no vectors to calibrate or store"),
            Self::CorruptedData(msg) => write!(f, "corrupted data block: {msg}"),
        }
    }
}

impl StdError for CompactError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::IoError { source } => Some(source),
            _ => None,
        }
    }
}

impl From<io::Error> for CompactError {
    fn from(source: io::Error) -> Self {
        Self::IoError { source }
    }
}

impl CompactError {
    /// Helper: construct `InvalidHeader` without `format!` at call-site repetition.
    #[inline]
    pub fn invalid_header(msg: impl Into<String>) -> Self {
        Self::InvalidHeader(msg.into())
    }

    /// Helper: construct `CorruptedFooter`.
    #[inline]
    pub fn corrupted_footer(msg: impl Into<String>) -> Self {
        Self::CorruptedFooter(msg.into())
    }

    /// Returns `true` if this is an I/O error.
    #[inline]
    pub fn is_io(&self) -> bool {
        matches!(self, Self::IoError { .. })
    }
}
