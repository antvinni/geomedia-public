//! ISO base media file format (MP4, MOV, M4A, 3GP).

pub mod meta;
pub mod walk;

pub use meta::{audio_priming_s, extract_bmff};
pub use walk::{BoxHeader, MAX_DEPTH, MAX_PAYLOAD, Walker};
