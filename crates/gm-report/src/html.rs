//! A single self-contained HTML page: inline CSS, no scripts, no external requests.

use std::fmt::Write;

use gm_media::{HintValue, MediaReport};

use crate::{format_name, kind_name, osm_link, stream_line, value_text};

/// Escape `& < > " '` for HTML text and attribute values.
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

const STYLE: &str = r#"
:root { --bg: #fbfaf8; --fg: #1d1d1b; --muted: #6b6a66; --line: #e3e1dc; --card: #ffffff; --accent: #2f5d8a; --warn: #9a5b00; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --fg: #ecebe8; --muted: #a09e99; --line: #2e2d2b; --card: #1e1e1c; --accent: #8ab4dc; --warn: #e0a54a; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; word-break: break-all; }
h2 { font-size: 13px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 28px 0 8px; }
.sub { color: var(--muted); margin: 0; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 13px; }
tr:last-child td { border-bottom: 0; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; word-break: break-all; }
a { color: var(--accent); }
.empty { font-weight: 600; }
.warn li { color: var(--warn); }
footer { color: var(--muted); margin-top: 32px; font-size: 13px; }
"#;

/// The full report page.
pub fn render_html(r: &MediaReport) -> String {
    let e = escape_html;
    let name = std::path::Path::new(&r.file.path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| r.file.path.clone());
    let mut o = String::new();
    let _ = write!(
        o,
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <meta name=\"color-scheme\" content=\"light dark\">\n<title>geomedia — {}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n<main>\n",
        e(&name)
    );
    let _ = writeln!(
        o,
        "<h1>{}</h1>\n<p class=\"sub\">geomedia metadata report · format: {}</p>",
        e(&name),
        format_name(r.format)
    );

    let _ = writeln!(o, "<h2>File</h2>\n<section><table>");
    let _ = writeln!(
        o,
        "<tr><th>Path</th><td class=\"mono\">{}</td></tr>",
        e(&r.file.path)
    );
    let _ = writeln!(
        o,
        "<tr><th>Size</th><td>{} bytes</td></tr>",
        r.file.size_bytes
    );
    let _ = writeln!(
        o,
        "<tr><th>SHA-256</th><td class=\"mono\">{}</td></tr>",
        e(&r.file.sha256)
    );
    if let Some(m) = &r.file.fs_modified_utc {
        let _ = writeln!(
            o,
            "<tr><th>Filesystem modified</th><td>{} <span class=\"sub\">(not embedded metadata)</span></td></tr>",
            e(m)
        );
    }
    let _ = writeln!(o, "</table></section>");

    if let Some(c) = &r.container
        && (c.major_brand.is_some() || c.duration_s.is_some())
    {
        let _ = writeln!(o, "<h2>Container</h2>\n<section><table>");
        if let Some(b) = &c.major_brand {
            let _ = writeln!(o, "<tr><th>Brand</th><td class=\"mono\">{}</td></tr>", e(b));
            if !c.compatible_brands.is_empty() {
                let _ = writeln!(
                    o,
                    "<tr><th>Compatible</th><td class=\"mono\">{}</td></tr>",
                    e(&c.compatible_brands.join(", "))
                );
            }
        }
        if let Some(d) = c.duration_s {
            let _ = writeln!(o, "<tr><th>Duration</th><td>{d:.3} s</td></tr>");
        }
        let _ = writeln!(o, "</table></section>");
    }

    if !r.streams.is_empty() {
        let _ = writeln!(o, "<h2>Streams</h2>\n<section><table>");
        for s in &r.streams {
            let _ = writeln!(o, "<tr><td class=\"mono\">{}</td></tr>", e(&stream_line(s)));
        }
        let _ = writeln!(o, "</table></section>");
    }

    let _ = writeln!(o, "<h2>Metadata (untrusted)</h2>\n<section>");
    if r.is_metadata_empty() {
        let _ = writeln!(o, "<p class=\"empty\">Metadata: empty</p>");
    } else {
        let _ = writeln!(
            o,
            "<table>\n<tr><th>Kind</th><th>Value</th><th>Source</th><th>Trust</th></tr>"
        );
        for h in &r.hints {
            let value = match h.value {
                HintValue::Location {
                    lat_deg, lon_deg, ..
                } => format!(
                    "{} · <a href=\"{}\">map</a>",
                    e(&value_text(&h.value)),
                    e(&osm_link(lat_deg, lon_deg))
                ),
                _ => e(&value_text(&h.value)),
            };
            let _ = writeln!(
                o,
                "<tr><td>{}</td><td>{value}</td><td class=\"mono\">{}</td><td>untrusted</td></tr>",
                kind_name(h.kind),
                e(&h.source)
            );
        }
        let _ = writeln!(o, "</table>");
    }
    let _ = writeln!(o, "</section>");

    if !r.warnings.is_empty() {
        let _ = writeln!(o, "<h2>Warnings</h2>\n<section class=\"warn\"><ul>");
        for w in &r.warnings {
            let _ = writeln!(o, "<li>{}</li>", e(w));
        }
        let _ = writeln!(o, "</ul></section>");
    }
    let _ = writeln!(
        o,
        "<footer>All embedded metadata is untrusted: it can be stripped, rewritten or forged.</footer>\n</main>\n</body>\n</html>"
    );
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use gm_media::{FileInfo, Hint, HintKind};
    use std::path::PathBuf;

    fn report(hints: Vec<Hint>) -> MediaReport {
        let mut r = MediaReport::new(FileInfo {
            path: "clip.mov".into(),
            ..FileInfo::default()
        });
        r.hints = hints;
        r
    }

    #[test]
    fn html_escapes_script() {
        let out = render_html(&report(vec![Hint::text(
            HintKind::Other,
            "tag:x",
            "<script>alert(1)</script>",
        )]));
        assert!(
            out.contains("&lt;script&gt;alert(1)&lt;/script&gt;"),
            "{out}"
        );
        assert!(!out.contains("<script>"));
        assert_eq!(
            escape_html(r#"a&b<c>"d"'e"#),
            "a&amp;b&lt;c&gt;&quot;d&quot;&#39;e"
        );
    }

    #[test]
    fn html_empty_line() {
        let out = render_html(&report(Vec::new()));
        assert!(out.contains("Metadata: empty"), "{out}");
        assert!(out.contains("<title>geomedia — clip.mov</title>"), "{out}");
        assert!(out.contains(
            "All embedded metadata is untrusted: it can be stripped, rewritten or forged."
        ));
    }

    #[test]
    fn html_no_external() {
        let out = render_html(&report(Vec::new()));
        assert!(!out.contains("src=\"http"));
        assert!(!out.contains("<script"));
        assert!(!out.contains("<link"));
        assert!(out.contains(r#"<meta name="color-scheme" content="light dark">"#));
    }

    #[test]
    fn html_phone_gps() {
        let p =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/phone_gps.mov");
        let mut r = gm_media::inspect(&p).unwrap();
        r.file.path = "phone_gps.mov".into();
        r.file.sha256 = "<redacted>".into();
        r.file.fs_modified_utc = None;
        insta::assert_snapshot!(render_html(&r));
    }
}
