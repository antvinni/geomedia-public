//! Metadata from MP4/MOV files: movie header times, tracks, udta atoms, iTunes and
//! QuickTime (mdta) item lists, and platform traces.

use std::io::{Read, Seek};

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::bmff::walk::{BoxHeader, Walker, fourcc_str};
use crate::iso6709::parse_iso6709;
use crate::model::{ContainerInfo, Extracted, Hint, HintKind, HintValue, StreamInfo, StreamKind};
use crate::traces::classify_trace;
use crate::when::{plausible, time_hint};

/// Seconds between 1904-01-01 (the QuickTime epoch) and 1970-01-01.
const QT_EPOCH_OFFSET: i64 = 2_082_844_800;
/// Small metadata boxes are read whole up to this size.
const META_MAX: usize = 64 * 1024;

/// Everything a BMFF file says about itself.
pub fn extract_bmff<R: Read + Seek>(r: R, file_len: u64) -> Extracted {
    let mut x = Extractor {
        w: Walker::new(r, file_len),
        hints: Vec::new(),
        warnings: Vec::new(),
    };
    let mut container = ContainerInfo::default();
    let mut streams = Vec::new();
    for b in x.w.children(0, file_len, 0) {
        match &b.fourcc {
            b"ftyp" => x.ftyp(&b, &mut container),
            b"moov" => {
                let (s, e) = x.w.child_range(&b);
                for k in x.w.children(s, e, 1) {
                    match &k.fourcc {
                        b"mvhd" => x.mvhd(&k, &mut container),
                        b"trak" => streams.push(x.trak(&k)),
                        b"udta" => x.udta(&k),
                        b"meta" => x.meta(&k, "moov/meta"),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    x.traces(&container, &streams);
    let Extractor {
        w,
        mut hints,
        mut warnings,
    } = x;
    let mut all = w.into_warnings();
    all.append(&mut warnings);
    hints.dedup();
    (Some(container), streams, hints, all)
}

/// Encoder priming of the first audio track, from its edit list (`edts/elst` media_time), in
/// seconds. AAC in MP4 typically starts 1024 samples in; decoders that ignore the edit list are
/// late by this much.
pub fn audio_priming_s<R: Read + Seek>(r: R, file_len: u64) -> Option<f64> {
    let mut x = Extractor {
        w: Walker::new(r, file_len),
        hints: Vec::new(),
        warnings: Vec::new(),
    };
    let moov =
        x.w.children(0, file_len, 0)
            .into_iter()
            .find(|b| &b.fourcc == b"moov")?;
    let (s, e) = x.w.child_range(&moov);
    for trak in
        x.w.children(s, e, 1)
            .into_iter()
            .filter(|b| &b.fourcc == b"trak")
    {
        let Some(hdlr) = x.find_in(&trak, &[*b"mdia", *b"hdlr"], 2) else {
            continue;
        };
        if x.w
            .payload(&hdlr, 512)
            .as_deref()
            .and_then(parse_hdlr)
            .map(|(k, _)| k)
            != Some(*b"soun")
        {
            continue;
        }
        let mdhd = x.find_in(&trak, &[*b"mdia", *b"mdhd"], 2)?;
        let timescale =
            x.w.payload(&mdhd, 64)
                .as_deref()
                .and_then(header_times)?
                .timescale;
        let elst = x.find_in(&trak, &[*b"edts", *b"elst"], 2)?;
        let p = x.w.payload(&elst, 4096)?;
        // version/flags(4) count(4), then entries: v0 (u32 duration, i32 media_time, rate) or v1 (u64, i64, rate).
        let media_time = if p.first() == Some(&1) {
            be_u64(&p, 16).map(|v| v as i64)?
        } else {
            i64::from(be_u32(&p, 12)? as i32)
        };
        return (media_time > 0 && timescale > 0).then(|| media_time as f64 / f64::from(timescale));
    }
    None
}

struct Extractor<R: Read + Seek> {
    w: Walker<R>,
    hints: Vec<Hint>,
    warnings: Vec<String>,
}

impl<R: Read + Seek> Extractor<R> {
    fn ftyp(&mut self, b: &BoxHeader, c: &mut ContainerInfo) {
        let Some(p) = self.w.payload(b, 1024) else {
            return;
        };
        if p.len() >= 4 {
            c.major_brand = Some(latin1(&p[..4]));
        }
        let (brands, _) = p.get(8..).unwrap_or(&[]).as_chunks::<4>();
        c.compatible_brands = brands.iter().map(|b| latin1(b)).collect();
    }

    fn mvhd(&mut self, b: &BoxHeader, c: &mut ContainerInfo) {
        let Some(p) = self.w.payload(b, 128) else {
            return;
        };
        let Some(h) = header_times(&p) else {
            self.warnings.push("truncated mvhd".into());
            return;
        };
        c.duration_s = h.duration_s();
        if let Some(hint) = qt_time_hint(HintKind::CreationTime, "moov/mvhd", h.created) {
            self.hints.push(hint);
        }
        if h.modified != h.created
            && let Some(hint) = qt_time_hint(HintKind::ModificationTime, "moov/mvhd", h.modified)
        {
            self.hints.push(hint);
        }
    }

    fn trak(&mut self, t: &BoxHeader) -> StreamInfo {
        let mut s = StreamInfo::new(StreamKind::Other);
        let (ts, te) = self.w.child_range(t);
        for k in self.w.children(ts, te, 2) {
            match &k.fourcc {
                b"tkhd" => self.tkhd(&k, &mut s),
                b"mdia" => {
                    let (ms, me) = self.w.child_range(&k);
                    for m in self.w.children(ms, me, 3) {
                        match &m.fourcc {
                            b"mdhd" => {
                                if let Some(h) =
                                    self.w.payload(&m, 64).as_deref().and_then(header_times)
                                {
                                    s.duration_s = h.duration_s();
                                }
                            }
                            b"hdlr" => {
                                if let Some((kind, name)) =
                                    self.w.payload(&m, 512).as_deref().and_then(parse_hdlr)
                                {
                                    s.kind = match &kind {
                                        b"soun" => StreamKind::Audio,
                                        b"vide" => StreamKind::Video,
                                        _ => StreamKind::Other,
                                    };
                                    s.handler_name = name;
                                }
                            }
                            b"minf" => {
                                if let Some(stsd) = self.find_in(&m, &[*b"stbl", *b"stsd"], 4) {
                                    self.stsd(&stsd, &mut s);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if s.kind != StreamKind::Video {
            s.width = None;
            s.height = None;
        }
        s
    }

    fn tkhd(&mut self, b: &BoxHeader, s: &mut StreamInfo) {
        let Some(p) = self.w.payload(b, 128) else {
            return;
        };
        let at = if p.first() == Some(&1) { 88 } else { 76 };
        let (Some(w), Some(h)) = (be_u32(&p, at), be_u32(&p, at + 4)) else {
            return;
        };
        if w > 0 && h > 0 {
            s.width = Some(w >> 16);
            s.height = Some(h >> 16);
        }
    }

    fn stsd(&mut self, b: &BoxHeader, s: &mut StreamInfo) {
        let Some(p) = self.w.payload(b, 512) else {
            return;
        };
        // version/flags(4) entry_count(4), then the first sample entry.
        let Some(e) = p.get(8..) else { return };
        if e.len() < 8 {
            return;
        }
        s.codec = Some(latin1(&e[4..8]));
        if s.kind == StreamKind::Audio {
            match be_u16(e, 16) {
                Some(2) => {
                    let rate = e
                        .get(40..48)
                        .and_then(|b| b.try_into().ok())
                        .map(f64::from_be_bytes);
                    s.sample_rate_hz = rate
                        .filter(|r| r.is_finite() && *r > 0.0 && *r < 1e7)
                        .map(|r| r.round() as u32);
                    s.channels = be_u32(e, 48).and_then(|c| u16::try_from(c).ok());
                }
                _ => {
                    s.channels = be_u16(e, 24);
                    s.sample_rate_hz = be_u32(e, 32).map(|r| r >> 16).filter(|r| *r > 0);
                }
            }
        }
    }

    fn udta(&mut self, u: &BoxHeader) {
        let (s, e) = self.w.child_range(u);
        for k in self.w.children(s, e, 2) {
            if k.fourcc == *b"meta" {
                self.meta(&k, "moov/udta/meta");
            } else if k.fourcc[0] == 0xA9 {
                let source = format!("moov/udta/{}", fourcc_str(&k.fourcc));
                let Some(p) = self.w.payload(&k, META_MAX) else {
                    continue;
                };
                let text = if p.get(4..8) == Some(b"data") {
                    parse_data(
                        p.get(8..be_u32(&p, 0).unwrap_or(0) as usize)
                            .unwrap_or(&p[8..]),
                    )
                } else {
                    qt_text(&p)
                };
                if let Some(text) = text {
                    self.item(&fourcc_str(&k.fourcc), source, text);
                }
            }
        }
    }

    /// A `meta` box: iTunes-style (`mdir`) item names, or QuickTime `mdta` keys.
    fn meta(&mut self, m: &BoxHeader, path: &str) {
        let (s, e) = self.w.child_range(m);
        let kids = self.w.children(s, e, 3);
        let mut handler = [0u8; 4];
        let mut keys: Vec<String> = Vec::new();
        for k in &kids {
            match &k.fourcc {
                b"hdlr" => {
                    if let Some((h, _)) = self.w.payload(k, 512).as_deref().and_then(parse_hdlr) {
                        handler = h;
                    }
                }
                b"keys" => {
                    keys = self
                        .w
                        .payload(k, META_MAX)
                        .map(|p| parse_keys(&p))
                        .unwrap_or_default()
                }
                _ => {}
            }
        }
        let Some(ilst) = kids.iter().find(|k| k.fourcc == *b"ilst") else {
            return;
        };
        let (is, ie) = self.w.child_range(ilst);
        for item in self.w.children(is, ie, 4) {
            let (name, source) = if &handler == b"mdta" {
                let idx = u32::from_be_bytes(item.fourcc) as usize;
                match idx.checked_sub(1).and_then(|i| keys.get(i)) {
                    Some(k) => (k.clone(), format!("mdta:{k}")),
                    None => {
                        self.warnings
                            .push(format!("{path}: ilst item {idx} has no key"));
                        continue;
                    }
                }
            } else {
                let n = fourcc_str(&item.fourcc);
                (n.clone(), format!("{path}/ilst/{n}"))
            };
            let Some(data) = self.find_in(&item, &[*b"data"], 5) else {
                continue;
            };
            if let Some(text) = self
                .w
                .payload(&data, META_MAX)
                .as_deref()
                .and_then(parse_data)
            {
                self.item(&name, source, text);
            }
        }
    }

    /// Map one named metadata item to a hint.
    fn item(&mut self, name: &str, source: String, text: String) {
        let kind = match name {
            "©xyz" | "com.apple.quicktime.location.ISO6709" | "location" => {
                let hint = match parse_iso6709(&text) {
                    Some((lat_deg, lon_deg, alt_m)) => Hint::new(
                        HintKind::Location,
                        source,
                        text,
                        HintValue::Location {
                            lat_deg,
                            lon_deg,
                            alt_m,
                        },
                    ),
                    None => {
                        self.warnings.push(format!(
                            "unparseable ISO 6709 location at {source}: {text:?}"
                        ));
                        Hint::text(HintKind::Location, source, text)
                    }
                };
                self.hints.push(hint);
                return;
            }
            "©day" | "com.apple.quicktime.creationdate" | "creation_time" | "date" => {
                self.hints
                    .push(time_hint(HintKind::CreationTime, &source, &text));
                return;
            }
            "©mak" | "©mod" | "com.apple.quicktime.make" | "com.apple.quicktime.model" => {
                HintKind::Device
            }
            "©swr" | "com.apple.quicktime.software" => HintKind::Software,
            "©too" | "©enc" | "encoder" => HintKind::Encoder,
            _ => HintKind::Other,
        };
        self.hints.push(Hint::text(kind, source, text));
    }

    /// Platform fingerprints from brands, handler names and software/encoder strings.
    fn traces(&mut self, c: &ContainerInfo, streams: &[StreamInfo]) {
        let mut found: Vec<(String, String, String)> = Vec::new();
        if let Some(b) = &c.major_brand {
            found.push(("ftyp".into(), b.clone(), b.trim().to_string()));
        }
        for (i, s) in streams.iter().enumerate() {
            if let Some(n) = &s.handler_name {
                found.push(("hdlr".into(), n.clone(), format!("trak[{i}]/mdia/hdlr")));
            }
        }
        for h in &self.hints {
            if matches!(h.kind, HintKind::Software | HintKind::Encoder) {
                found.push((h.source.clone(), h.raw.clone(), h.source.clone()));
            }
        }
        let mut seen: Vec<String> = Vec::new();
        for (kind_src, text, at) in found {
            let probe = if kind_src == "ftyp" {
                "ftyp"
            } else {
                kind_src.as_str()
            };
            if let Some(label) = classify_trace(probe, &text)
                && !seen.contains(&label)
            {
                let source = if kind_src == "ftyp" {
                    "ftyp".to_string()
                } else {
                    at
                };
                seen.push(label.clone());
                self.hints.push(Hint::new(
                    HintKind::PlatformTrace,
                    source,
                    text,
                    HintValue::Text { text: label },
                ));
            }
        }
    }

    /// Follow `path` below container `b`.
    fn find_in(&mut self, b: &BoxHeader, path: &[[u8; 4]], depth: u8) -> Option<BoxHeader> {
        let mut cur = *b;
        for (i, want) in path.iter().enumerate() {
            let (s, e) = self.w.child_range(&cur);
            let d = depth.saturating_add(u8::try_from(i).unwrap_or(u8::MAX));
            cur = self
                .w
                .children(s, e, d)
                .into_iter()
                .find(|k| &k.fourcc == want)?;
        }
        Some(cur)
    }
}

/// Creation, modification, timescale and duration from an mvhd/mdhd payload (v0 or v1).
struct HeaderTimes {
    created: u64,
    modified: u64,
    timescale: u32,
    duration: u64,
}

impl HeaderTimes {
    fn duration_s(&self) -> Option<f64> {
        (self.timescale > 0
            && self.duration > 0
            && self.duration != u64::from(u32::MAX)
            && self.duration != u64::MAX)
            .then(|| self.duration as f64 / f64::from(self.timescale))
    }
}

fn header_times(p: &[u8]) -> Option<HeaderTimes> {
    if p.first() == Some(&1) {
        Some(HeaderTimes {
            created: be_u64(p, 4)?,
            modified: be_u64(p, 12)?,
            timescale: be_u32(p, 20)?,
            duration: be_u64(p, 24)?,
        })
    } else {
        Some(HeaderTimes {
            created: u64::from(be_u32(p, 4)?),
            modified: u64::from(be_u32(p, 8)?),
            timescale: be_u32(p, 12)?,
            duration: u64::from(be_u32(p, 16)?),
        })
    }
}

/// A time hint for QuickTime-epoch seconds; 0 means "not set".
fn qt_time_hint(kind: HintKind, source: &str, secs_since_1904: u64) -> Option<Hint> {
    if secs_since_1904 == 0 {
        return None;
    }
    let unix = i64::try_from(secs_since_1904).ok()? - QT_EPOCH_OFFSET;
    let t = OffsetDateTime::from_unix_timestamp(unix).ok()?;
    Some(Hint::new(
        kind,
        source,
        format!("{secs_since_1904} s since 1904-01-01"),
        HintValue::Time {
            utc: t.format(&Rfc3339).ok()?,
            plausible: plausible(t),
        },
    ))
}

/// Handler type and name from an hdlr payload. QuickTime names are Pascal strings, ISO names NUL-terminated.
fn parse_hdlr(p: &[u8]) -> Option<([u8; 4], Option<String>)> {
    let kind: [u8; 4] = p.get(8..12)?.try_into().ok()?;
    let raw = p.get(24..).unwrap_or(&[]);
    let name = match raw.first() {
        None => None,
        Some(&n)
            if (n as usize) < raw.len() && raw[1..=n as usize].iter().all(|c| *c != 0) && n > 0 =>
        {
            Some(String::from_utf8_lossy(&raw[1..=n as usize]).into_owned())
        }
        Some(_) => {
            let end = raw.iter().position(|c| *c == 0).unwrap_or(raw.len());
            Some(String::from_utf8_lossy(&raw[..end]).into_owned())
        }
    };
    Some((
        kind,
        name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()),
    ))
}

/// Key names from a QuickTime `keys` payload.
fn parse_keys(p: &[u8]) -> Vec<String> {
    let count = be_u32(p, 4).unwrap_or(0) as usize;
    let mut out = Vec::new();
    let mut at = 8usize;
    while out.len() < count {
        let Some(size) = be_u32(p, at).map(|s| s as usize) else {
            break;
        };
        if size < 8 {
            break;
        }
        let Some(name) = p.get(at + 8..at + size) else {
            break;
        };
        out.push(String::from_utf8_lossy(name).into_owned());
        at += size;
    }
    out
}

/// The value of a `data` atom payload: type(4) locale(4) value.
fn parse_data(p: &[u8]) -> Option<String> {
    let ty = be_u32(p, 0)? & 0x00FF_FFFF;
    let v = p.get(8..)?;
    let s = match ty {
        1 => String::from_utf8_lossy(v).into_owned(),
        2 => {
            let (units, _) = v.as_chunks::<2>();
            String::from_utf16_lossy(
                &units
                    .iter()
                    .map(|u| u16::from_be_bytes(*u))
                    .collect::<Vec<_>>(),
            )
        }
        21 | 22 if !v.is_empty() && v.len() <= 8 => {
            let mut n: u64 = 0;
            for b in v {
                n = (n << 8) | u64::from(*b);
            }
            n.to_string()
        }
        _ => return None,
    };
    let s = s.trim_end_matches('\0').to_string();
    (!s.is_empty()).then_some(s)
}

/// A QuickTime udta text atom: length(2) language(2) text.
fn qt_text(p: &[u8]) -> Option<String> {
    let len = be_u16(p, 0)? as usize;
    let text = p.get(4..4 + len).or_else(|| p.get(4..))?;
    let s = String::from_utf8_lossy(text)
        .trim_end_matches('\0')
        .to_string();
    (!s.is_empty()).then_some(s)
}

fn latin1(b: &[u8]) -> String {
    b.iter()
        .map(|&c| if c == 0xA9 { '©' } else { c as char })
        .collect()
}

fn be_u16(p: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(p.get(at..at + 2)?.try_into().ok()?))
}

fn be_u32(p: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(p.get(at..at + 4)?.try_into().ok()?))
}

fn be_u64(p: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(p.get(at..at + 8)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bmff::walk::tests::bx;
    use crate::model::{HintKind, HintValue, StreamKind};
    use std::fs::File;
    use std::io::Cursor;
    use std::path::PathBuf;

    fn fixture(
        name: &str,
    ) -> (
        Option<ContainerInfo>,
        Vec<StreamInfo>,
        Vec<Hint>,
        Vec<String>,
    ) {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        let f = File::open(&p).unwrap();
        let len = f.metadata().unwrap().len();
        extract_bmff(f, len)
    }

    fn bytes(
        b: Vec<u8>,
    ) -> (
        Option<ContainerInfo>,
        Vec<StreamInfo>,
        Vec<Hint>,
        Vec<String>,
    ) {
        let len = b.len() as u64;
        extract_bmff(Cursor::new(b), len)
    }

    fn of(hints: &[Hint], kind: HintKind) -> Vec<&Hint> {
        hints.iter().filter(|h| h.kind == kind).collect()
    }

    fn loc(h: &Hint) -> (f64, f64, Option<f64>) {
        match h.value {
            HintValue::Location {
                lat_deg,
                lon_deg,
                alt_m,
            } => (lat_deg, lon_deg, alt_m),
            _ => panic!("not a location: {h:?}"),
        }
    }

    #[test]
    fn phone_gps_mov() {
        let (c, streams, hints, warnings) = fixture("phone_gps.mov");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(c.unwrap().major_brand.as_deref(), Some("qt  "));
        let locs = of(&hints, HintKind::Location);
        assert_eq!(locs.len(), 1, "{hints:?}");
        assert_eq!(locs[0].source, "mdta:com.apple.quicktime.location.ISO6709");
        let (lat, lon, alt) = loc(locs[0]);
        assert!((lat - 51.5072).abs() < 1e-6 && (lon + 0.1276).abs() < 1e-6);
        assert_eq!(alt, Some(11.0));
        let devices: Vec<_> = of(&hints, HintKind::Device)
            .iter()
            .map(|h| h.raw.clone())
            .collect();
        assert_eq!(devices, ["Apple", "iPhone 15"]);
        let created = of(&hints, HintKind::CreationTime);
        assert_eq!(created.len(), 1);
        assert_eq!(
            created[0].value,
            HintValue::Time {
                utc: "2026-05-14T10:02:00-07:00".into(),
                plausible: true
            }
        );
        let video: Vec<_> = streams
            .iter()
            .filter(|s| s.kind == StreamKind::Video)
            .collect();
        let audio: Vec<_> = streams
            .iter()
            .filter(|s| s.kind == StreamKind::Audio)
            .collect();
        assert_eq!((video.len(), audio.len()), (1, 1));
        assert_eq!((video[0].width, video[0].height), (Some(64), Some(64)));
        assert_eq!(video[0].codec.as_deref(), Some("mp4v"));
        assert_eq!(audio[0].sample_rate_hz, Some(48000));
        assert_eq!(audio[0].channels, Some(1));
        assert_eq!(audio[0].codec.as_deref(), Some("mp4a"));
        assert!(!of(&hints, HintKind::PlatformTrace).is_empty());
    }

    #[test]
    fn aac_priming_from_edit_list() {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/onset.m4a");
        let f = File::open(&p).unwrap();
        let len = f.metadata().unwrap().len();
        let s = audio_priming_s(f, len).unwrap();
        assert!((s - 1024.0 / 8000.0).abs() < 1e-9, "{s}");
    }

    #[test]
    fn phone_udta_mov() {
        let (_, _, hints, _) = fixture("phone_udta.mov");
        let locs = of(&hints, HintKind::Location);
        assert_eq!(locs.len(), 1, "{hints:?}");
        assert_eq!(locs[0].source, "moov/udta/©xyz");
        let (lat, lon, alt) = loc(locs[0]);
        assert!((lat - 51.5072).abs() < 1e-6 && (lon + 0.1276).abs() < 1e-6);
        assert_eq!(alt, None);
        assert!(
            of(&hints, HintKind::Software)
                .iter()
                .any(|h| h.raw.starts_with("Lavf"))
        );
    }

    #[test]
    fn reencoded_mp4() {
        let (c, _, hints, _) = fixture("reencoded.mp4");
        assert_eq!(c.unwrap().compatible_brands, ["isom", "iso2", "mp41"]);
        assert!(of(&hints, HintKind::Location).is_empty());
        assert!(of(&hints, HintKind::CreationTime).is_empty());
        let traces = of(&hints, HintKind::PlatformTrace);
        assert!(
            traces
                .iter()
                .any(|h| matches!(&h.value, HintValue::Text { text } if text.contains("ffmpeg")))
        );
    }

    fn mvhd_v0(creation: u32) -> Vec<u8> {
        let mut p = vec![0u8; 4];
        p.extend_from_slice(&creation.to_be_bytes());
        p.extend_from_slice(&creation.to_be_bytes());
        p.extend_from_slice(&1000u32.to_be_bytes());
        p.extend_from_slice(&5000u32.to_be_bytes());
        p.resize(100, 0);
        bx(b"mvhd", &p)
    }

    #[test]
    fn plain_qt_is_metadata_empty() {
        let (c, _, hints, _) =
            bytes([bx(b"ftyp", b"qt  \0\0\0\0qt  "), bx(b"moov", &mvhd_v0(0))].concat());
        assert_eq!(c.unwrap().major_brand.as_deref(), Some("qt  "));
        assert!(hints.is_empty(), "{hints:?}");
    }

    #[test]
    fn mvhd_zero_time_absent() {
        let (c, _, hints, _) =
            bytes([bx(b"ftyp", b"isom\0\0\0\0"), bx(b"moov", &mvhd_v0(0))].concat());
        assert!(of(&hints, HintKind::CreationTime).is_empty());
        assert_eq!(c.unwrap().duration_s, Some(5.0));
    }

    #[test]
    fn mvhd_1970_confusion_flagged() {
        let (_, _, hints, _) =
            bytes([bx(b"ftyp", b"isom\0\0\0\0"), bx(b"moov", &mvhd_v0(86_400))].concat());
        let created = of(&hints, HintKind::CreationTime);
        assert_eq!(created.len(), 1);
        assert!(
            matches!(
                created[0].value,
                HintValue::Time {
                    plausible: false,
                    ..
                }
            ),
            "{created:?}"
        );
    }

    #[test]
    fn mvhd_real_time_plausible() {
        // 2026-05-14T17:02:00Z in seconds since 1904-01-01.
        let secs = (1_778_778_120u64 + 2_082_844_800) as u32;
        let (_, _, hints, _) =
            bytes([bx(b"ftyp", b"isom\0\0\0\0"), bx(b"moov", &mvhd_v0(secs))].concat());
        let created = of(&hints, HintKind::CreationTime);
        assert_eq!(
            created[0].value,
            HintValue::Time {
                utc: "2026-05-14T17:02:00Z".into(),
                plausible: true
            }
        );
        assert_eq!(created[0].source, "moov/mvhd");
    }

    #[test]
    fn moov_meta_apple_location() {
        // iPhone layout: moov/meta (QuickTime style, no version) with hdlr mdta, keys, ilst.
        let mut hdlr = vec![0u8; 8];
        hdlr.extend_from_slice(b"mdta");
        hdlr.resize(25, 0);
        let key = b"com.apple.quicktime.location.ISO6709";
        let mut keys = vec![0, 0, 0, 0, 0, 0, 0, 1];
        keys.extend_from_slice(&((key.len() + 8) as u32).to_be_bytes());
        keys.extend_from_slice(b"mdta");
        keys.extend_from_slice(key);
        let mut data = vec![0, 0, 0, 1, 0, 0, 0, 0];
        data.extend_from_slice(b"+37.7749-122.4194+010.000/");
        let item = bx(&1u32.to_be_bytes(), &bx(b"data", &data));
        let meta = bx(
            b"meta",
            &[bx(b"hdlr", &hdlr), bx(b"keys", &keys), bx(b"ilst", &item)].concat(),
        );
        let (_, _, hints, w) = bytes([bx(b"ftyp", b"qt  \0\0\0\0"), bx(b"moov", &meta)].concat());
        assert!(w.is_empty(), "{w:?}");
        let locs = of(&hints, HintKind::Location);
        assert_eq!(locs.len(), 1, "{hints:?}");
        assert_eq!(loc(locs[0]), (37.7749, -122.4194, Some(10.0)));
    }

    #[test]
    fn bad_iso6709_kept_as_text() {
        let xyz = bx(b"\xa9xyz", b"\0\x07\x15\xc7garbage");
        let (_, _, hints, w) = bytes(
            [
                bx(b"ftyp", b"qt  \0\0\0\0"),
                bx(b"moov", &bx(b"udta", &xyz)),
            ]
            .concat(),
        );
        let locs = of(&hints, HintKind::Location);
        assert_eq!(locs.len(), 1);
        assert_eq!(
            locs[0].value,
            HintValue::Text {
                text: "garbage".into()
            }
        );
        assert!(!w.is_empty());
    }
}
