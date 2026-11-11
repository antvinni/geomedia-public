//! Renderers for geomedia reports: terminal text, JSON and self-contained HTML.

pub mod html;
pub mod json;
pub mod text;

pub use html::{escape_html, render_html};
pub use json::render_json;
pub use text::{format_name, render_text, sanitize_terminal, stream_line};

use gm_media::{HintKind, HintValue};

/// The snake_case name used for a hint kind in every output.
pub fn kind_name(k: HintKind) -> &'static str {
    match k {
        HintKind::Location => "location",
        HintKind::CreationTime => "creation_time",
        HintKind::ModificationTime => "modification_time",
        HintKind::Device => "device",
        HintKind::Software => "software",
        HintKind::Encoder => "encoder",
        HintKind::PlatformTrace => "platform_trace",
        HintKind::Other => "other",
    }
}

/// An OpenStreetMap link centred on a point.
pub fn osm_link(lat: f64, lon: f64) -> String {
    format!("https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon}")
}

/// A hint's value as display text (unsanitised).
pub fn value_text(v: &HintValue) -> String {
    match v {
        HintValue::Location {
            lat_deg,
            lon_deg,
            alt_m,
        } => match alt_m {
            Some(a) => format!("{lat_deg:.6}, {lon_deg:.6}, {a} m"),
            None => format!("{lat_deg:.6}, {lon_deg:.6}"),
        },
        HintValue::Time { utc, plausible } => {
            if *plausible {
                utc.clone()
            } else {
                format!("{utc} (implausible)")
            }
        }
        HintValue::Text { text } => text.clone(),
    }
}
