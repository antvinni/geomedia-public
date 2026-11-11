//! The terminal summary. Every file-derived string passes through `sanitize_terminal`.

use std::fmt::Write;

use gm_media::{HintValue, MediaReport, StreamKind};

use crate::{kind_name, osm_link, value_text};

/// Longest string printed for any single value.
pub const MAX_CHARS: usize = 200;

/// Make `s` safe to print: control characters become visible `\xNN` escapes, bidi controls `\u{..}`, long values are cut.
pub fn sanitize_terminal(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i == MAX_CHARS {
            out.push('…');
            break;
        }
        let code = c as u32;
        if c != '\t' && (code < 0x20 || (0x7f..=0x9f).contains(&code)) {
            let _ = write!(out, "\\x{code:02x}");
        } else if is_bidi_control(c) {
            let _ = write!(out, "\\u{{{code:x}}}");
        } else {
            out.push(c);
        }
    }
    out
}

/// Unicode bidirectional controls, which can visually reorder the text around them.
fn is_bidi_control(c: char) -> bool {
    matches!(c, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
}

/// The human-readable summary printed by `geomedia meta`.
pub fn render_text(r: &MediaReport) -> String {
    let s = sanitize_terminal;
    let mut o = String::new();
    let _ = writeln!(o, "File");
    let _ = writeln!(o, "  path: {}", s(&r.file.path));
    let _ = writeln!(o, "  size: {} bytes", r.file.size_bytes);
    let _ = writeln!(o, "  sha256: {}", r.file.sha256);
    let _ = writeln!(o, "Format");
    let _ = writeln!(o, "  {}", format_name(r.format));
    if let Some(c) = &r.container {
        let mut lines = Vec::new();
        if let Some(b) = &c.major_brand {
            let compat = if c.compatible_brands.is_empty() {
                String::new()
            } else {
                format!(" (compatible: {})", c.compatible_brands.join(", "))
            };
            lines.push(format!("  brand: {}{}", s(b), s(&compat)));
        }
        if let Some(d) = c.duration_s {
            lines.push(format!("  duration: {d:.3} s"));
        }
        if !lines.is_empty() {
            let _ = writeln!(o, "Container");
            for l in lines {
                let _ = writeln!(o, "{l}");
            }
        }
    }
    if !r.streams.is_empty() {
        let _ = writeln!(o, "Streams");
        for st in &r.streams {
            let _ = writeln!(o, "  {}", s(&stream_line(st)));
        }
    }
    if r.is_metadata_empty() {
        let _ = writeln!(o, "metadata: empty");
    } else {
        let _ = writeln!(o, "Metadata (untrusted)");
        for h in &r.hints {
            let mut line = format!(
                "  {}  {}  [{}]",
                kind_name(h.kind),
                s(&value_text(&h.value)),
                s(&h.source)
            );
            if let HintValue::Location {
                lat_deg, lon_deg, ..
            } = h.value
            {
                let _ = write!(line, "  {}", osm_link(lat_deg, lon_deg));
            }
            let _ = writeln!(o, "{line}");
        }
    }
    if !r.warnings.is_empty() {
        let _ = writeln!(o, "Warnings");
        for w in &r.warnings {
            let _ = writeln!(o, "  {}", s(w));
        }
    }
    o
}

/// One-line description of a stream, e.g. "audio mp4a 48000 Hz 1 ch 1.000 s".
pub fn stream_line(st: &gm_media::StreamInfo) -> String {
    let mut parts = vec![match st.kind {
        StreamKind::Audio => "audio".to_string(),
        StreamKind::Video => "video".to_string(),
        StreamKind::Other => "other".to_string(),
    }];
    if let Some(c) = &st.codec {
        parts.push(c.clone());
    }
    if let (Some(w), Some(h)) = (st.width, st.height) {
        parts.push(format!("{w}x{h}"));
    }
    if let Some(r) = st.sample_rate_hz {
        parts.push(format!("{r} Hz"));
    }
    if let Some(c) = st.channels {
        parts.push(format!("{c} ch"));
    }
    if let Some(d) = st.duration_s {
        parts.push(format!("{d:.3} s"));
    }
    if let Some(n) = &st.handler_name {
        parts.push(format!("[{n}]"));
    }
    parts.join(" ")
}

/// The snake_case format name, or "unrecognised".
pub fn format_name(f: gm_media::Format) -> &'static str {
    use gm_media::Format::*;
    match f {
        Bmff => "bmff",
        Wav => "wav",
        Mp3 => "mp3",
        Flac => "flac",
        Ogg => "ogg",
        Matroska => "matroska",
        Aiff => "aiff",
        Jpeg => "jpeg",
        Png => "png",
        Heif => "heif",
        Unrecognised => "unrecognised",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gm_media::{FileInfo, Hint, HintKind, HintValue, MediaReport};

    fn report(hints: Vec<Hint>) -> MediaReport {
        let mut r = MediaReport::new(FileInfo {
            path: "clip.mov".into(),
            size_bytes: 3,
            sha256: "ab".into(),
            fs_modified_utc: None,
        });
        r.hints = hints;
        r
    }

    #[test]
    fn text_empty() {
        let out = render_text(&report(Vec::new()));
        assert!(out.lines().any(|l| l == "metadata: empty"), "{out}");
        assert!(out.contains("File"), "{out}");
        assert!(out.lines().any(|l| l == "Format"), "{out}");
    }

    #[test]
    fn text_sanitizes() {
        let out = render_text(&report(vec![Hint::text(
            HintKind::Other,
            "tag:x",
            "\x1b[2Jevil",
        )]));
        assert!(out.contains("\\x1b[2Jevil"), "{out}");
        assert!(!out.contains('\x1b'));
        assert_eq!(sanitize_terminal("a\tb\x07\u{9b}c"), "a\tb\\x07\\x9bc");
        let long = "y".repeat(500);
        assert_eq!(sanitize_terminal(&long).chars().count(), MAX_CHARS + 1);
        assert!(sanitize_terminal(&long).ends_with('…'));
    }

    #[test]
    fn text_escapes_bidi_overrides() {
        assert_eq!(sanitize_terminal("51.5\u{202e}7"), "51.5\\u{202e}7");
        assert_eq!(
            sanitize_terminal("a\u{2066}b\u{200f}c"),
            "a\\u{2066}b\\u{200f}c"
        );
    }

    #[test]
    fn text_location_osm_link() {
        let h = Hint::new(
            HintKind::Location,
            "moov/udta/©xyz",
            "+51.5072-000.1276/",
            HintValue::Location {
                lat_deg: 51.5072,
                lon_deg: -0.1276,
                alt_m: None,
            },
        );
        let out = render_text(&report(vec![h]));
        assert!(out.contains("Metadata (untrusted)"), "{out}");
        assert!(out.contains("51.507200, -0.127600"), "{out}");
        assert!(out.contains("mlat=51.5072&mlon=-0.1276"), "{out}");
        assert!(!out.contains("metadata: empty"));
    }

    #[test]
    fn text_flags_implausible_time() {
        let h = Hint::new(
            HintKind::CreationTime,
            "moov/mvhd",
            "86400 s since 1904-01-01",
            HintValue::Time {
                utc: "1904-01-02T00:00:00Z".into(),
                plausible: false,
            },
        );
        let out = render_text(&report(vec![h]));
        assert!(out.contains("1904-01-02T00:00:00Z (implausible)"), "{out}");
    }
}
