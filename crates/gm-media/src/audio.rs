//! Audio containers (WAV, MP3, FLAC, Ogg, Matroska, AIFF) through symphonia, plus the
//! Broadcast WAV `bext` chunk, which symphonia does not expose.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use symphonia::core::codecs::CodecParameters;
use symphonia::core::formats::probe::Hint as ProbeHint;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::{MetadataOptions, MetadataRevision, RawValue, StandardTag, Tag};

use crate::iso6709::parse_iso6709;
use crate::model::{
    ContainerInfo, Extracted, Format, Hint, HintKind, HintValue, StreamInfo, StreamKind,
};
use crate::traces::classify_trace;
use crate::when::time_hint;

/// Container, streams, tag hints and warnings for a non-BMFF audio container.
pub fn extract_audio(path: &Path, format: Format) -> Extracted {
    let mut warnings = Vec::new();
    let (container, streams, mut hints) = match catch_unwind(AssertUnwindSafe(|| probe(path))) {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => {
            warnings.push(format!("audio container could not be parsed: {e}"));
            (None, Vec::new(), Vec::new())
        }
        Err(_) => {
            warnings
                .push("audio container parser crashed on this file; results are partial".into());
            (None, Vec::new(), Vec::new())
        }
    };
    if format == Format::Wav {
        match read_bext(path) {
            Ok(mut h) => hints.append(&mut h),
            Err(e) => warnings.push(format!("could not scan RIFF chunks: {e}")),
        }
    }
    if format == Format::Matroska && streams.iter().all(|s| s.kind == StreamKind::Audio) {
        warnings.push("video tracks in Matroska not listed in v0".into());
    }
    let traces: Vec<Hint> = hints
        .iter()
        .filter(|h| matches!(h.kind, HintKind::Encoder | HintKind::Software))
        .filter_map(|h| {
            classify_trace(&h.source, &h.raw).map(|label| {
                Hint::new(
                    HintKind::PlatformTrace,
                    h.source.clone(),
                    h.raw.clone(),
                    HintValue::Text { text: label },
                )
            })
        })
        .collect();
    for t in traces {
        if !hints
            .iter()
            .any(|h| h.kind == HintKind::PlatformTrace && h.value == t.value)
        {
            hints.push(t);
        }
    }
    (container, streams, hints, warnings)
}

type Probed = (Option<ContainerInfo>, Vec<StreamInfo>, Vec<Hint>);

fn probe(path: &Path) -> Result<Probed, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
    let mut reader = symphonia::default::get_probe()
        .probe(
            &ProbeHint::new(),
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| e.to_string())?;
    let streams: Vec<StreamInfo> = reader.tracks().iter().map(stream_of).collect();
    let duration_s = streams
        .iter()
        .filter_map(|s| s.duration_s)
        .fold(None, |a: Option<f64>, d| Some(a.map_or(d, |a| a.max(d))));
    let mut hints = Vec::new();
    collect_metadata(reader.as_mut(), &mut hints);
    let container = ContainerInfo {
        major_brand: None,
        compatible_brands: Vec::new(),
        duration_s,
    };
    Ok((Some(container), streams, hints))
}

fn stream_of(t: &symphonia::core::formats::Track) -> StreamInfo {
    let mut s = StreamInfo::new(StreamKind::Other);
    match &t.codec_params {
        Some(CodecParameters::Audio(a)) => {
            s.kind = StreamKind::Audio;
            s.sample_rate_hz = a.sample_rate;
            s.channels = a
                .channels
                .as_ref()
                .and_then(|c| u16::try_from(c.count()).ok());
            s.codec = Some(
                symphonia::default::get_codecs()
                    .get_audio_decoder(a.codec)
                    .map(|d| d.codec.info.short_name.to_string())
                    .unwrap_or_else(|| a.codec.to_string()),
            );
        }
        Some(CodecParameters::Video(v)) => {
            s.kind = StreamKind::Video;
            s.width = v.width.map(u32::from);
            s.height = v.height.map(u32::from);
            s.codec = Some(v.codec.to_string());
        }
        _ => {}
    }
    if let (Some(tb), Some(d)) = (t.time_base, t.duration) {
        s.duration_s = tb.calc_duration(d).map(|t| t.as_secs_f64());
    }
    s
}

fn collect_metadata(reader: &mut dyn FormatReader, hints: &mut Vec<Hint>) {
    let mut log = reader.metadata();
    let mut revisions: Vec<MetadataRevision> = Vec::new();
    while let Some(rev) = log.pop() {
        revisions.push(rev);
    }
    if let Some(rev) = log.current() {
        revisions.push(rev.clone());
    }
    for rev in &revisions {
        for tag in rev
            .media
            .tags
            .iter()
            .chain(rev.per_track.iter().flat_map(|t| t.metadata.tags.iter()))
        {
            if let Some(h) = tag_hint(tag)
                && !hints.contains(&h)
            {
                hints.push(h);
            }
        }
    }
}

fn tag_hint(tag: &Tag) -> Option<Hint> {
    let text = raw_text(&tag.raw.value)?;
    let source = format!("tag:{}", tag.raw.key);
    let key = tag.raw.key.to_ascii_lowercase();
    let kind = match &tag.std {
        Some(StandardTag::RecordingDate(_) | StandardTag::EncodingDate(_)) => {
            HintKind::CreationTime
        }
        Some(
            StandardTag::Encoder(_) | StandardTag::EncodedBy(_) | StandardTag::EncoderSettings(_),
        ) => HintKind::Encoder,
        Some(StandardTag::RecordingLocation(_)) => HintKind::Location,
        _ => match key.as_str() {
            "date" | "tdrc" | "tdat" | "tyer" | "icrd" | "creation_time" | "year" => {
                HintKind::CreationTime
            }
            "encoder" | "encoded_by" | "tenc" | "tsse" | "isft" => HintKind::Encoder,
            "location" | "©xyz" => HintKind::Location,
            _ => HintKind::Other,
        },
    };
    Some(match kind {
        HintKind::CreationTime => time_hint(kind, &source, &text),
        HintKind::Location => match parse_iso6709(&text) {
            Some((lat_deg, lon_deg, alt_m)) => Hint::new(
                kind,
                source,
                text,
                HintValue::Location {
                    lat_deg,
                    lon_deg,
                    alt_m,
                },
            ),
            None => Hint::text(kind, source, text),
        },
        _ => Hint::text(kind, source, text),
    })
}

fn raw_text(v: &RawValue) -> Option<String> {
    let s = match v {
        RawValue::String(s) => s.to_string(),
        RawValue::StringList(l) => l.join("; "),
        RawValue::SignedInt(n) => n.to_string(),
        RawValue::UnsignedInt(n) => n.to_string(),
        RawValue::Float(f) => f.to_string(),
        RawValue::Boolean(b) => b.to_string(),
        _ => return None,
    };
    let s = s.trim().to_string();
    (!s.is_empty()).then_some(s)
}

/// Size of the fixed part of a `bext` chunk that we read.
const BEXT_LEN: usize = 338;

/// Origination date/time and originator from a Broadcast WAV `bext` chunk (EBU Tech 3285).
fn read_bext(path: &Path) -> std::io::Result<Vec<Hint>> {
    let mut f = File::open(path)?;
    let len = f.metadata()?.len();
    let mut at = 12u64;
    while at + 8 <= len {
        f.seek(SeekFrom::Start(at))?;
        let mut h = [0u8; 8];
        f.read_exact(&mut h)?;
        let size = u64::from(u32::from_le_bytes([h[4], h[5], h[6], h[7]]));
        if &h[..4] == b"bext" {
            if (size as usize) < BEXT_LEN {
                return Ok(Vec::new());
            }
            let mut b = vec![0u8; BEXT_LEN];
            f.read_exact(&mut b)?;
            return Ok(bext_hints(&b));
        }
        at += 8 + size + (size & 1);
    }
    Ok(Vec::new())
}

fn bext_hints(b: &[u8]) -> Vec<Hint> {
    let field = |r: std::ops::Range<usize>| -> String {
        let raw = &b[r];
        let end = raw.iter().position(|c| *c == 0).unwrap_or(raw.len());
        String::from_utf8_lossy(&raw[..end]).trim().to_string()
    };
    let mut out = Vec::new();
    let originator = field(256..288);
    let date = field(320..330);
    let time = field(330..338);
    if !date.is_empty() {
        // EBU allows any separator; normalise to ISO 8601.
        let norm = |s: &str, sep: char, at: [usize; 2]| -> String {
            s.chars()
                .enumerate()
                .map(|(i, c)| if at.contains(&i) { sep } else { c })
                .collect()
        };
        let d = norm(&date, '-', [4, 7]);
        let t = if time.is_empty() {
            String::new()
        } else {
            format!(" {}", norm(&time, ':', [2, 5]))
        };
        let mut h = time_hint(HintKind::CreationTime, "riff:bext", &format!("{d}{t}"));
        h.raw = format!("{d}{t} (local time, no zone)");
        out.push(h);
    }
    if !originator.is_empty() {
        out.push(Hint::text(HintKind::Device, "riff:bext", originator));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{HintKind, HintValue, StreamKind};
    use std::path::PathBuf;

    fn fixture(
        name: &str,
        format: Format,
    ) -> (
        Option<ContainerInfo>,
        Vec<StreamInfo>,
        Vec<Hint>,
        Vec<String>,
    ) {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        extract_audio(&p, format)
    }

    fn assert_tagged(name: &str, format: Format) {
        let (c, streams, hints, warnings) = fixture(name, format);
        assert!(warnings.is_empty(), "{name}: {warnings:?}");
        let created: Vec<_> = hints
            .iter()
            .filter(|h| h.kind == HintKind::CreationTime)
            .collect();
        assert_eq!(created.len(), 1, "{name}: {hints:?}");
        assert_eq!(
            created[0].value,
            HintValue::Time {
                utc: "2026-05-14".into(),
                plausible: true
            }
        );
        assert!(
            hints
                .iter()
                .any(|h| h.kind == HintKind::Encoder && h.raw == "TestRig"),
            "{name}: {hints:?}"
        );
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].kind, StreamKind::Audio);
        assert_eq!(streams[0].sample_rate_hz, Some(48000));
        assert_eq!(streams[0].channels, Some(1));
        let d = c
            .and_then(|c| c.duration_s)
            .or(streams[0].duration_s)
            .expect("duration");
        assert!((d - 1.0).abs() < 0.1, "{name}: duration {d}");
    }

    #[test]
    fn mp3_id3_tags() {
        assert_tagged("tagged.mp3", Format::Mp3);
    }

    #[test]
    fn flac_vorbis_comment() {
        assert_tagged("tagged.flac", Format::Flac);
    }

    #[test]
    fn bwf_origination() {
        let (_, streams, hints, warnings) = fixture("bwf.wav", Format::Wav);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(streams.len(), 1);
        let created: Vec<_> = hints
            .iter()
            .filter(|h| h.kind == HintKind::CreationTime)
            .collect();
        assert_eq!(created.len(), 1, "{hints:?}");
        assert_eq!(created[0].source, "riff:bext");
        assert!(
            created[0].raw.contains("2026-05-14 10:02:00"),
            "{}",
            created[0].raw
        );
        assert_eq!(
            created[0].value,
            HintValue::Time {
                utc: "2026-05-14T10:02:00".into(),
                plausible: true
            }
        );
        assert!(
            hints
                .iter()
                .any(|h| h.kind == HintKind::Device && h.raw == "TestRig")
        );
    }

    #[test]
    fn plain_wav_empty() {
        let (_, streams, hints, warnings) = fixture("plain.wav", Format::Wav);
        assert!(hints.is_empty(), "{hints:?}");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].kind, StreamKind::Audio);
        assert_eq!(streams[0].sample_rate_hz, Some(8000));
    }
}
