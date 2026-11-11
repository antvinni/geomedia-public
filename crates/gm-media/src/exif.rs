//! EXIF from still images (JPEG, PNG, HEIF): GPS, capture time, device, software.

use std::io::{BufRead, Seek};

use ::exif::{Exif, Field, In, Tag, Value};

use crate::model::{Hint, HintKind, HintValue};
use crate::when::time_hint;

/// Hints and warnings from an image's EXIF block. A missing block is not a warning.
pub fn extract_exif<R: BufRead + Seek>(mut r: R) -> (Vec<Hint>, Vec<String>) {
    let exif = match ::exif::Reader::new().read_from_container(&mut r) {
        Ok(e) => e,
        Err(::exif::Error::NotFound(_)) => return (Vec::new(), Vec::new()),
        Err(e) => {
            return (
                Vec::new(),
                vec![format!("EXIF block could not be parsed: {e}")],
            );
        }
    };
    let mut hints = Vec::new();
    if let Some(h) = gps(&exif) {
        hints.push(h);
    }
    if let Some(dt) = ascii(&exif, Tag::DateTimeOriginal) {
        let offset = ascii(&exif, Tag::OffsetTimeOriginal);
        hints.push(capture_time(
            &dt,
            offset.as_deref(),
            "exif:DateTimeOriginal",
        ));
    } else if let Some(dt) = ascii(&exif, Tag::DateTime) {
        let offset = ascii(&exif, Tag::OffsetTime);
        hints.push(capture_time(&dt, offset.as_deref(), "exif:DateTime"));
    }
    for (tag, kind) in [
        (Tag::Make, HintKind::Device),
        (Tag::Model, HintKind::Device),
        (Tag::Software, HintKind::Software),
    ] {
        if let Some(v) = ascii(&exif, tag) {
            hints.push(Hint::text(kind, format!("exif:{tag}"), v));
        }
    }
    (hints, Vec::new())
}

/// "2026:05:14 10:02:00" plus an optional "-07:00" offset; without one the time is local.
fn capture_time(dt: &str, offset: Option<&str>, source: &str) -> Hint {
    let iso: String = dt
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if (i == 4 || i == 7) && c == ':' {
                '-'
            } else {
                c
            }
        })
        .collect();
    let (input, raw) = match offset {
        Some(o) => (format!("{iso}{o}"), format!("{dt} {o}")),
        None => (iso, format!("{dt} (local time, no zone)")),
    };
    let mut h = time_hint(HintKind::CreationTime, source, &input);
    h.raw = raw;
    h
}

fn gps(exif: &Exif) -> Option<Hint> {
    let lat = dms(exif.get_field(Tag::GPSLatitude, In::PRIMARY)?)?;
    let lon = dms(exif.get_field(Tag::GPSLongitude, In::PRIMARY)?)?;
    let lat = if ascii(exif, Tag::GPSLatitudeRef).as_deref() == Some("S") {
        -lat
    } else {
        lat
    };
    let lon = if ascii(exif, Tag::GPSLongitudeRef).as_deref() == Some("W") {
        -lon
    } else {
        lon
    };
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return None;
    }
    let alt = exif
        .get_field(Tag::GPSAltitude, In::PRIMARY)
        .and_then(|f| match &f.value {
            Value::Rational(v) => v.first().filter(|r| r.denom != 0).map(|r| r.to_f64()),
            _ => None,
        });
    let below = matches!(
        exif.get_field(Tag::GPSAltitudeRef, In::PRIMARY).map(|f| &f.value),
        Some(Value::Byte(v)) if v.first() == Some(&1)
    );
    let alt_m = alt.map(|a| if below { -a } else { a });
    let raw = format!(
        "{lat:.7}, {lon:.7}{}",
        alt_m.map(|a| format!(", {a} m")).unwrap_or_default()
    );
    Some(Hint::new(
        HintKind::Location,
        "exif:GPS",
        raw,
        HintValue::Location {
            lat_deg: lat,
            lon_deg: lon,
            alt_m,
        },
    ))
}

/// Degrees from three rationals (degrees, minutes, seconds).
fn dms(f: &Field) -> Option<f64> {
    let Value::Rational(v) = &f.value else {
        return None;
    };
    if v.len() < 3 || v.iter().take(3).any(|r| r.denom == 0) {
        return None;
    }
    Some(v[0].to_f64() + v[1].to_f64() / 60.0 + v[2].to_f64() / 3600.0)
}

fn ascii(exif: &Exif, tag: Tag) -> Option<String> {
    let f = exif.get_field(tag, In::PRIMARY)?;
    let Value::Ascii(parts) = &f.value else {
        return None;
    };
    let s = parts
        .iter()
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    let s = s.trim_end_matches('\0').trim().to_string();
    (!s.is_empty()).then_some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{HintKind, HintValue};
    use std::fs::File;
    use std::io::BufReader;
    use std::path::PathBuf;

    fn fixture(name: &str) -> (Vec<Hint>, Vec<String>) {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        extract_exif(BufReader::new(File::open(p).unwrap()))
    }

    #[test]
    fn jpeg_gps() {
        let (hints, warnings) = fixture("gps.jpg");
        assert!(warnings.is_empty(), "{warnings:?}");
        let loc = hints
            .iter()
            .find(|h| h.kind == HintKind::Location)
            .expect("location");
        assert_eq!(loc.source, "exif:GPS");
        let HintValue::Location {
            lat_deg,
            lon_deg,
            alt_m,
        } = loc.value
        else {
            panic!("{loc:?}")
        };
        assert!((lat_deg - 51.5072).abs() < 1e-6, "{lat_deg}");
        assert!((lon_deg + 0.1276).abs() < 1e-6, "{lon_deg}");
        assert_eq!(alt_m, Some(11.0));
        let devices: Vec<_> = hints
            .iter()
            .filter(|h| h.kind == HintKind::Device)
            .map(|h| h.raw.as_str())
            .collect();
        assert_eq!(devices, ["TestCam", "T1"]);
        let created = hints
            .iter()
            .find(|h| h.kind == HintKind::CreationTime)
            .expect("time");
        assert_eq!(
            created.value,
            HintValue::Time {
                utc: "2026-05-14T10:02:00-07:00".into(),
                plausible: true
            }
        );
    }

    #[test]
    fn fallback_time_keeps_its_source() {
        let h = capture_time("2026:05:14 10:02:00", None, "exif:DateTime");
        assert_eq!(h.source, "exif:DateTime");
        assert_eq!(
            h.value,
            HintValue::Time {
                utc: "2026-05-14T10:02:00".into(),
                plausible: true
            }
        );
    }

    #[test]
    fn png_no_exif_empty() {
        let (hints, warnings) = fixture("nogps.png");
        assert!(hints.is_empty(), "{hints:?}");
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}
