//! The machine-readable report (schema version 1).

use gm_media::MediaReport;

/// Pretty JSON with a trailing newline.
pub fn render_json(r: &MediaReport) -> String {
    let mut s =
        serde_json::to_string_pretty(r).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"));
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use gm_media::FileInfo;

    #[test]
    fn json_roundtrip_fields() {
        let out = render_json(&MediaReport::new(FileInfo::default()));
        assert!(out.ends_with('\n'));
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["hints"], serde_json::json!([]));
    }
}
