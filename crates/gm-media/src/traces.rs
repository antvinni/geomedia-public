//! Recognise encoders and handlers that reveal how a file was produced or processed.

/// A human-readable label when `text` is a known platform fingerprint. `_source` (where it was
/// found) is kept for rules that depend on location.
pub fn classify_trace(_source: &str, text: &str) -> Option<String> {
    let t = text.trim();
    let label = if t.starts_with("Lavf")
        || t.starts_with("Lavc")
        || t == "VideoHandler"
        || t == "SoundHandler"
    {
        "ffmpeg/libavformat mux (file was processed or re-encoded)"
    } else if t.contains("Core Media") {
        "Apple Core Media (iPhone/macOS capture or export)"
    } else if t.contains("produced by Google") {
        "Google/Android or YouTube processing"
    } else {
        return None;
    };
    Some(label.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_lavf() {
        let t = classify_trace("encoder", "Lavf61.7.100").expect("lavf");
        assert!(t.contains("ffmpeg"), "{t}");
    }

    #[test]
    fn classify_known_handlers() {
        assert!(
            classify_trace("hdlr", "VideoHandler")
                .unwrap()
                .contains("ffmpeg")
        );
        assert!(
            classify_trace("hdlr", "Core Media Video")
                .unwrap()
                .contains("Apple")
        );
        assert!(
            classify_trace("hdlr", "ISO Media file produced by Google Inc.")
                .unwrap()
                .contains("Google")
        );
        // The QuickTime brand is a container fact, already shown as the brand; not a trace.
        assert_eq!(classify_trace("ftyp", "qt  "), None);
        assert_eq!(classify_trace("hdlr", "DataHandler"), None);
    }
}
