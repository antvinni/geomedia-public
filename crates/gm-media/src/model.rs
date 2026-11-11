//! The report every inspection produces. Embedded values are hints, never truth.

use serde::Serialize;

/// Version of the JSON report schema.
pub const SCHEMA_VERSION: u32 = 1;

/// What one extractor returns: container facts, streams, hints and warnings.
pub type Extracted = (
    Option<ContainerInfo>,
    Vec<StreamInfo>,
    Vec<Hint>,
    Vec<String>,
);

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct MediaReport {
    pub schema_version: u32,
    pub file: FileInfo,
    pub format: Format,
    pub container: Option<ContainerInfo>,
    pub streams: Vec<StreamInfo>,
    pub hints: Vec<Hint>,
    pub warnings: Vec<String>,
}

impl MediaReport {
    /// A report for `file` with nothing extracted yet.
    pub fn new(file: FileInfo) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            file,
            format: Format::Unrecognised,
            container: None,
            streams: Vec::new(),
            hints: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// True when the file carries no embedded metadata at all.
    pub fn is_metadata_empty(&self) -> bool {
        self.hints.is_empty()
    }
}

#[derive(Serialize, Debug, Clone, PartialEq, Default)]
pub struct FileInfo {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    /// Filesystem modification time (RFC 3339). Not embedded metadata.
    pub fs_modified_utc: Option<String>,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Bmff,
    Wav,
    Mp3,
    Flac,
    Ogg,
    Matroska,
    Aiff,
    Jpeg,
    Png,
    Heif,
    Unrecognised,
}

#[derive(Serialize, Debug, Clone, PartialEq, Default)]
pub struct ContainerInfo {
    pub major_brand: Option<String>,
    pub compatible_brands: Vec<String>,
    pub duration_s: Option<f64>,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StreamKind {
    Audio,
    Video,
    Other,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct StreamInfo {
    pub kind: StreamKind,
    pub codec: Option<String>,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration_s: Option<f64>,
    pub handler_name: Option<String>,
}

impl StreamInfo {
    /// A stream of `kind` with every detail unknown.
    pub fn new(kind: StreamKind) -> Self {
        Self {
            kind,
            codec: None,
            sample_rate_hz: None,
            channels: None,
            width: None,
            height: None,
            duration_s: None,
            handler_name: None,
        }
    }
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HintKind {
    Location,
    CreationTime,
    ModificationTime,
    Device,
    Software,
    Encoder,
    PlatformTrace,
    Other,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HintValue {
    Location {
        lat_deg: f64,
        lon_deg: f64,
        alt_m: Option<f64>,
    },
    Time {
        /// RFC 3339; without an offset when the source recorded local time.
        utc: String,
        plausible: bool,
    },
    Text {
        text: String,
    },
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    Untrusted,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct Hint {
    pub kind: HintKind,
    /// Where it came from, e.g. "moov/udta/©xyz" or "exif:GPS".
    pub source: String,
    pub raw: String,
    pub value: HintValue,
    pub trust: Trust,
}

impl Hint {
    pub fn new(
        kind: HintKind,
        source: impl Into<String>,
        raw: impl Into<String>,
        value: HintValue,
    ) -> Self {
        Self {
            kind,
            source: source.into(),
            raw: raw.into(),
            value,
            trust: Trust::Untrusted,
        }
    }

    /// A plain-text hint whose value is its raw string.
    pub fn text(kind: HintKind, source: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        Self::new(kind, source, text.clone(), HintValue::Text { text })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_json_shape() {
        let r = MediaReport::new(FileInfo::default());
        let j = serde_json::to_string(&r).unwrap();
        assert!(j.contains(r#""schema_version":1"#), "{j}");
        assert!(j.contains(r#""hints":[]"#), "{j}");
        assert!(j.contains(r#""format":"unrecognised""#), "{j}");
        assert!(r.is_metadata_empty());
    }
}
