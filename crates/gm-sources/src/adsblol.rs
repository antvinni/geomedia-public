//! adsb.lol `globe_history` day archives: a split tar of per-aircraft readsb traces
//! (`traces/<xx>/trace_full_<icao>.json`, gzip despite the name). One streaming pass builds a
//! small index (offset, size, bounding box, time span per trace), so later queries seek
//! straight to the traces they need.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::SourceError;

/// Largest decompressed trace accepted (untrusted archive).
pub const MAX_TRACE_BYTES: u64 = 64 << 20;

pub struct Day {
    pub date: time::Date,
    pub parts: Vec<PathBuf>,
}

/// Read + Seek over the concatenation of the archive's parts.
pub struct MultiPart {
    parts: Vec<(File, u64)>,
    pos: u64,
    total: u64,
}

impl MultiPart {
    pub fn open(paths: &[PathBuf]) -> Result<MultiPart, SourceError> {
        let mut parts = Vec::with_capacity(paths.len());
        let mut total = 0;
        for p in paths {
            let f = File::open(p).map_err(|source| SourceError::Io {
                path: p.display().to_string(),
                source,
            })?;
            let len = f
                .metadata()
                .map_err(|source| SourceError::Io {
                    path: p.display().to_string(),
                    source,
                })?
                .len();
            total += len;
            parts.push((f, len));
        }
        Ok(MultiPart {
            parts,
            pos: 0,
            total,
        })
    }

    pub fn len(&self) -> u64 {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }
}

impl Read for MultiPart {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut start = 0;
        for (f, len) in &mut self.parts {
            if self.pos < start + *len {
                let within = self.pos - start;
                f.seek(SeekFrom::Start(within))?;
                let n = (*len - within).min(buf.len() as u64) as usize;
                let got = f.read(&mut buf[..n])?;
                self.pos += got as u64;
                return Ok(got);
            }
            start += *len;
        }
        Ok(0)
    }
}

impl Seek for MultiPart {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let target = match to {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::End(d) => i128::from(self.total) + i128::from(d),
            SeekFrom::Current(d) => i128::from(self.pos) + i128::from(d),
        };
        if target < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start",
            ));
        }
        self.pos = target as u64;
        Ok(self.pos)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEntry {
    pub icao: String,
    /// Byte offset of the entry's data in the concatenated archive.
    pub offset: u64,
    pub len: u64,
    /// UNIX seconds.
    pub t0: f64,
    pub t1: f64,
    pub lat_min: f64,
    pub lat_max: f64,
    /// When `lon_min > lon_max` the box wraps across the antimeridian.
    pub lon_min: f64,
    pub lon_max: f64,
    pub type_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayIndex {
    pub date: String,
    pub entries: Vec<TraceEntry>,
    pub skipped: usize,
}

/// Release assets (name, download URL, size) of a GitHub releases API response.
pub fn parse_release_assets(json: &str) -> Result<Vec<(String, String, u64)>, SourceError> {
    #[derive(Deserialize)]
    struct Asset {
        name: String,
        browser_download_url: String,
        size: u64,
    }
    #[derive(Deserialize)]
    struct Release {
        assets: Vec<Asset>,
    }
    let r: Release = serde_json::from_str(json)
        .map_err(|e| SourceError::Invalid(format!("unexpected releases API response: {e}")))?;
    let mut v: Vec<_> = r
        .assets
        .into_iter()
        .filter(|a| a.name.contains(".tar"))
        .map(|a| (a.name, a.browser_download_url, a.size))
        .collect();
    v.sort();
    Ok(v)
}

/// The archive parts for `date` (GitHub releases of adsblol/globe_history_<year>).
pub fn day_assets(date: time::Date) -> Result<Vec<(String, String, u64)>, SourceError> {
    let d = date_str(date).replace('-', ".");
    let url = format!(
        "https://api.github.com/repos/adsblol/globe_history_{}/releases/tags/v{d}-planes-readsb-prod-0",
        date.year()
    );
    let resp = crate::cache::agent()
        .get(&url)
        .call()
        .map_err(|e| SourceError::Http {
            url: url.clone(),
            reason: e.to_string(),
        })?;
    match resp.status().as_u16() {
        200 => {}
        404 => {
            return Err(SourceError::Invalid(format!(
                "adsb.lol has no archive for {} (yet)",
                date_str(date)
            )));
        }
        s => {
            return Err(SourceError::Http {
                url,
                reason: format!("HTTP {s}"),
            });
        }
    }
    let text = resp
        .into_body()
        .read_to_string()
        .map_err(|e| SourceError::Http {
            url,
            reason: e.to_string(),
        })?;
    parse_release_assets(&text)
}

/// The day's archive, downloading it if absent (needs `consent`: it is several GB).
pub fn ensure_day(
    date: time::Date,
    consent: bool,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<Day, SourceError> {
    let dir = day_dir(date);
    if let Some(parts) = local_parts(&dir) {
        return Ok(Day { date, parts });
    }
    let assets = day_assets(date)?;
    let total: u64 = assets.iter().map(|a| a.2).sum();
    if !consent {
        return Err(SourceError::NeedsConsent {
            what: format!("ADS-B for {}", date_str(date)),
            size: crate::human_size(total),
        });
    }
    write_manifest(
        &dir,
        &assets
            .iter()
            .map(|a| (a.0.clone(), a.2))
            .collect::<Vec<_>>(),
    )?;
    let _ = std::fs::remove_file(dir.join("index.json"));
    let mut parts = Vec::new();
    for (name, url, size) in assets {
        let f = crate::Fetch {
            url,
            dest: dir.join(&name),
            expected_len: Some(size),
            timeout: None,
        };
        parts.push(crate::fetch(&f, &mut |done, total| {
            progress(&name, done, total)
        })?);
    }
    Ok(Day { date, parts })
}

/// The day's index, building it on first use.
pub fn day_index(day: &Day) -> Result<DayIndex, SourceError> {
    crate::paths::touch_used("adsb", &date_str(day.date));
    index_parts(day.date, &day.parts, &day_dir(day.date))
}

/// The day's cache directory.
pub fn day_dir(date: time::Date) -> PathBuf {
    crate::cache_dir().join("adsb").join(date_str(date))
}

pub fn date_str(d: time::Date) -> String {
    format!("{:04}-{:02}-{:02}", d.year(), u8::from(d.month()), d.day())
}

/// Record the expected parts and sizes of a day (written before downloading).
pub fn write_manifest(dir: &Path, parts: &[(String, u64)]) -> Result<(), SourceError> {
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    let m: Vec<Manifest> = parts
        .iter()
        .map(|(name, size)| Manifest {
            name: name.clone(),
            size: *size,
        })
        .collect();
    let path = dir.join(MANIFEST);
    std::fs::write(
        &path,
        serde_json::to_vec(&m).map_err(|e| SourceError::Invalid(e.to_string()))?,
    )
    .map_err(io(&path))
}

const MANIFEST: &str = "manifest.json";

#[derive(Serialize, Deserialize)]
struct Manifest {
    name: String,
    size: u64,
}

/// Already-downloaded parts in `dir` (`*.tar` or `*.tar.aa`, `*.tar.ab`, …), if complete.
fn io(p: &Path) -> impl Fn(std::io::Error) -> SourceError + '_ {
    move |source| SourceError::Io {
        path: p.display().to_string(),
        source,
    }
}

pub fn local_parts(dir: &Path) -> Option<Vec<PathBuf>> {
    // With a manifest (written before downloading), every listed part must be present and complete.
    if let Ok(text) = std::fs::read(dir.join(MANIFEST)) {
        let m: Vec<Manifest> = serde_json::from_slice(&text).ok()?;
        let mut parts = Vec::new();
        for e in &m {
            let p = dir.join(&e.name);
            if std::fs::metadata(&p).ok()?.len() != e.size {
                return None;
            }
            parts.push(p);
        }
        parts.sort();
        return (!parts.is_empty()).then_some(parts);
    }
    // Legacy caches without a manifest: accept a set of parts with no partial download.
    let mut parts = Vec::new();
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.ends_with(".part") {
            return None;
        }
        let is_part = name.ends_with(".tar")
            || name.rsplit_once(".tar.").is_some_and(|(_, suf)| {
                suf.len() == 2 && suf.bytes().all(|b| b.is_ascii_lowercase())
            });
        if is_part {
            parts.push(e.path());
        }
    }
    parts.sort();
    (!parts.is_empty()).then_some(parts)
}

/// Index of the archive at `parts`; written to / read from `<dir>/index.json`.
pub fn index_parts(
    date: time::Date,
    parts: &[PathBuf],
    dir: &Path,
) -> Result<DayIndex, SourceError> {
    let cached = dir.join("index.json");
    if let Ok(text) = std::fs::read_to_string(&cached)
        && let Ok(idx) = serde_json::from_str::<DayIndex>(&text)
        && idx.date == date_str(date)
    {
        return Ok(idx);
    }
    let mut archive = tar::Archive::new(MultiPart::open(parts)?);
    let (tx, rx) = std::sync::mpsc::sync_channel::<(String, u64, Vec<u8>)>(256);
    let rx = std::sync::Mutex::new(rx);
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 16);
    let mut entries: Vec<TraceEntry> = Vec::new();
    let mut skipped = 0usize;
    std::thread::scope(|scope| -> Result<(), SourceError> {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let rx = &rx;
                scope.spawn(move || {
                    let mut out = Vec::new();
                    let mut bad = 0usize;
                    loop {
                        let job = rx.lock().ok().and_then(|r| r.recv().ok());
                        let Some((icao, offset, data)) = job else {
                            break;
                        };
                        match summarize(&icao, offset, &data) {
                            Some(e) => out.push(e),
                            None => bad += 1,
                        }
                    }
                    (out, bad)
                })
            })
            .collect();
        let first = &parts[0];
        for entry in archive.entries().map_err(io(first))? {
            let mut entry = entry.map_err(io(first))?;
            let path = entry
                .path()
                .map_err(io(first))?
                .to_string_lossy()
                .into_owned();
            let Some(icao) = path
                .rsplit_once("trace_full_")
                .and_then(|(_, n)| n.strip_suffix(".json"))
                .map(str::to_string)
            else {
                continue;
            };
            let size = entry.size();
            if size > MAX_TRACE_BYTES {
                skipped += 1;
                continue;
            }
            let offset = entry.raw_file_position();
            let mut data = Vec::with_capacity(size as usize);
            entry.read_to_end(&mut data).map_err(io(first))?;
            if tx.send((icao, offset, data)).is_err() {
                break;
            }
        }
        drop(tx);
        for h in handles {
            if let Ok((mut out, bad)) = h.join() {
                entries.append(&mut out);
                skipped += bad;
            }
        }
        Ok(())
    })?;
    entries.sort_by_key(|e| e.offset);
    let idx = DayIndex {
        date: date_str(date),
        entries,
        skipped,
    };
    let tmp = dir.join("index.json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_vec(&idx).map_err(|e| SourceError::Invalid(e.to_string()))?,
    )
    .map_err(io(&tmp))?;
    std::fs::rename(&tmp, &cached).map_err(io(&cached))?;
    Ok(idx)
}

/// Gunzip (bounded) a trace entry's bytes; plain JSON is accepted too.
pub fn gunzip_bounded(data: &[u8]) -> Option<Vec<u8>> {
    if data.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(data)
            .take(MAX_TRACE_BYTES)
            .read_to_end(&mut out)
            .ok()?;
        Some(out)
    } else if data.first() == Some(&b'{') {
        Some(data.to_vec())
    } else {
        None
    }
}

/// The first three fields of a trace point (dt, lat, lon); the rest is skipped.
struct Head(f64, Option<f64>, Option<f64>);

impl<'de> Deserialize<'de> for Head {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Head;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a trace point array")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Head, A::Error> {
                let dt: f64 = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("empty point"))?;
                let lat: Option<f64> = seq.next_element()?.flatten();
                let lon: Option<f64> = seq.next_element()?.flatten();
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(Head(dt, lat, lon))
            }
        }
        d.deserialize_seq(V)
    }
}

#[derive(Deserialize)]
struct Summary {
    #[serde(default)]
    t: Option<String>,
    timestamp: f64,
    trace: Vec<Head>,
}

fn summarize(icao: &str, offset: u64, data: &[u8]) -> Option<TraceEntry> {
    let json = gunzip_bounded(data)?;
    let s: Summary = serde_json::from_slice(&json).ok()?;
    let mut e = TraceEntry {
        icao: icao.to_string(),
        offset,
        len: data.len() as u64,
        t0: f64::INFINITY,
        t1: f64::NEG_INFINITY,
        lat_min: f64::INFINITY,
        lat_max: f64::NEG_INFINITY,
        lon_min: f64::INFINITY,
        lon_max: f64::NEG_INFINITY,
        type_code: s.t.filter(|t| !t.is_empty()),
    };
    let mut lons = Vec::new();
    for Head(dt, lat, lon) in &s.trace {
        let (Some(lat), Some(lon)) = (lat, lon) else {
            continue;
        };
        if !(lat.is_finite() && lon.is_finite() && lat.abs() <= 90.0 && lon.abs() <= 180.0) {
            continue;
        }
        let t = s.timestamp + dt;
        e.t0 = e.t0.min(t);
        e.t1 = e.t1.max(t);
        e.lat_min = e.lat_min.min(*lat);
        e.lat_max = e.lat_max.max(*lat);
        lons.push(*lon);
    }
    if lons.is_empty() {
        return None;
    }
    let (lo, hi) = lon_extent(&lons);
    e.lon_min = lo;
    e.lon_max = hi;
    Some(e)
}

/// Longitude extent: the circle minus its largest empty gap. A box crossing the antimeridian
/// wraps (`min > max`).
pub fn lon_extent(lons: &[f64]) -> (f64, f64) {
    let mut v: Vec<f64> = lons.iter().copied().filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    let (Some(&first), Some(&last)) = (v.first(), v.last()) else {
        return (f64::NAN, f64::NAN);
    };
    let wrap_gap = first + 360.0 - last;
    let (mut best, mut at) = (wrap_gap, None);
    for (i, w) in v.windows(2).enumerate() {
        if w[1] - w[0] > best {
            best = w[1] - w[0];
            at = Some(i);
        }
    }
    match at {
        None => (first, last),
        Some(i) => (v[i + 1], v[i]),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    pub(crate) fn gz(s: &str) -> Vec<u8> {
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(s.as_bytes()).unwrap();
        e.finish().unwrap()
    }

    pub(crate) fn trace_json(icao: &str, base: f64, pts: &[(f64, f64, f64)]) -> String {
        // Fill in points every 20 s between the given ones (real cruise sampling).
        let mut dense = Vec::new();
        for w in pts.windows(2) {
            let ((t0, la0, lo0), (t1, la1, lo1)) = (w[0], w[1]);
            let n = ((t1 - t0) / 20.0).ceil().max(1.0) as usize;
            for i in 0..n {
                let f = i as f64 / n as f64;
                dense.push((
                    t0 + f * (t1 - t0),
                    la0 + f * (la1 - la0),
                    lo0 + f * (lo1 - lo0),
                ));
            }
        }
        dense.extend(pts.last().copied());
        let pts: Vec<String> = dense
            .iter()
            .map(|(dt, lat, lon)| {
                format!(
                    "[{dt},{lat},{lon},5000,120.0,90.0,0,0,null,\"adsb_icao\",5200,null,null,null]"
                )
            })
            .collect();
        format!(
            r#"{{"icao":"{icao}","r":"N1","t":"PC12","timestamp":{base},"trace":[{}]}}"#,
            pts.join(",")
        )
    }

    /// A day archive with three traces and one corrupt entry, split into .aa/.ab at `split`.
    pub(crate) fn fixture_day(dir: &Path, split: usize) -> Vec<PathBuf> {
        let mut b = tar::Builder::new(Vec::new());
        let mut add = |name: &str, data: &[u8]| {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            b.append_data(&mut h, name, data).unwrap();
        };
        add("./README.txt", b"readme");
        add(
            "./traces/eb/trace_full_a1dfeb.json",
            &gz(&trace_json(
                "a1dfeb",
                1_788_825_600.0,
                &[
                    (75000.0, 40.73, -73.77),
                    (75100.0, 40.80, -73.60),
                    (75200.0, 40.88, -73.40),
                ],
            )),
        );
        add(
            "./traces/91/trace_full_ace391.json",
            &gz(&trace_json(
                "ace391",
                1_788_825_600.0,
                &[(76000.0, 40.95, -73.30), (76300.0, 40.85, -73.55)],
            )),
        );
        add("./traces/01/trace_full_bad001.json", b"not gzip at all");
        add(
            "./traces/02/trace_full_aa0002.json",
            &gz(&trace_json(
                "aa0002",
                1_788_825_600.0,
                &[(1000.0, 51.47, -0.45), (1100.0, 51.50, -0.30)],
            )),
        );
        let bytes = b.into_inner().unwrap();
        let (a, c) = bytes.split_at(split.min(bytes.len()));
        let pa = dir.join("v2026.09.08-planes-readsb-prod-0.tar.aa");
        let pb = dir.join("v2026.09.08-planes-readsb-prod-0.tar.ab");
        std::fs::write(&pa, a).unwrap();
        std::fs::write(&pb, c).unwrap();
        vec![pa, pb]
    }

    fn date() -> time::Date {
        time::Date::from_calendar_date(2026, time::Month::September, 8).unwrap()
    }

    #[test]
    fn multipart_reads_across_split() {
        let dir = tempfile::tempdir().unwrap();
        let parts = fixture_day(dir.path(), 1777);
        let whole: Vec<u8> = parts
            .iter()
            .flat_map(|p| std::fs::read(p).unwrap())
            .collect();
        let mut m = MultiPart::open(&parts).unwrap();
        assert_eq!(m.len(), whole.len() as u64);
        let mut buf = Vec::new();
        m.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, whole);
        m.seek(SeekFrom::Start(1700)).unwrap();
        let mut chunk = vec![0u8; 200];
        m.read_exact(&mut chunk).unwrap();
        assert_eq!(chunk, whole[1700..1900]);
    }

    #[test]
    fn index_entries() {
        let dir = tempfile::tempdir().unwrap();
        let parts = fixture_day(dir.path(), 2049);
        assert_eq!(local_parts(dir.path()), Some(parts.clone()));
        let idx = index_parts(date(), &parts, dir.path()).unwrap();
        assert_eq!(idx.entries.len(), 3, "{idx:?}");
        assert_eq!(idx.skipped, 1);
        let e = idx.entries.iter().find(|e| e.icao == "a1dfeb").unwrap();
        assert_eq!(e.type_code.as_deref(), Some("PC12"));
        assert!(
            (e.t0 - (1_788_825_600.0 + 75000.0)).abs() < 1e-6
                && (e.t1 - (1_788_825_600.0 + 75200.0)).abs() < 1e-6
        );
        assert!((e.lat_min - 40.73).abs() < 1e-9 && (e.lon_max + 73.40).abs() < 1e-9);
        // The offset reads back the exact gz bytes.
        let mut m = MultiPart::open(&parts).unwrap();
        m.seek(SeekFrom::Start(e.offset)).unwrap();
        let mut data = vec![0u8; e.len as usize];
        m.read_exact(&mut data).unwrap();
        assert_eq!(&data[..2], &[0x1f, 0x8b]);
        // A second call reads the cached index.
        assert!(dir.path().join("index.json").exists());
        assert_eq!(
            index_parts(date(), &parts, dir.path()).unwrap().entries,
            idx.entries
        );
    }

    #[test]
    fn manifest_rejects_incomplete_day() {
        let dir = tempfile::tempdir().unwrap();
        let parts = fixture_day(dir.path(), 2049);
        let sizes: Vec<(String, u64)> = parts
            .iter()
            .map(|p| {
                (
                    p.file_name().unwrap().to_string_lossy().into_owned(),
                    std::fs::metadata(p).unwrap().len(),
                )
            })
            .collect();
        write_manifest(dir.path(), &sizes).unwrap();
        assert_eq!(local_parts(dir.path()), Some(parts.clone()));
        // Truncated part.
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&parts[1])
            .unwrap();
        f.set_len(sizes[1].1 - 512).unwrap();
        assert_eq!(local_parts(dir.path()), None);
        // Missing part.
        std::fs::remove_file(&parts[1]).unwrap();
        assert_eq!(local_parts(dir.path()), None);
    }

    #[test]
    fn lon_extent_wide_and_wrapping() {
        assert_eq!(lon_extent(&[-120.0, -60.0, 0.0, 77.0]), (-120.0, 77.0));
        assert_eq!(lon_extent(&[179.5, 179.9, -179.8, -179.2]), (179.5, -179.2));
        assert_eq!(
            lon_extent(&[-170.0, -160.0, 160.0, 170.0]),
            (160.0, -160.0),
            "the largest gap (-160..160) is excluded"
        );
        assert_eq!(lon_extent(&[5.0]), (5.0, 5.0));
    }

    #[test]
    fn parses_release_assets() {
        let json = r#"{"tag_name":"v2026.09.08-planes-readsb-prod-0","assets":[
            {"name":"v2026.09.08-planes-readsb-prod-0.tar.ab","browser_download_url":"https://x/ab","size":5},
            {"name":"v2026.09.08-planes-readsb-prod-0.tar.aa","browser_download_url":"https://x/aa","size":7},
            {"name":"notes.txt","browser_download_url":"https://x/n","size":1}]}"#;
        let a = parse_release_assets(json).unwrap();
        assert_eq!(a.len(), 2);
        assert!(a[0].0.ends_with(".aa") && a[0].2 == 7);
        assert!(parse_release_assets("{}").is_err());
    }

    #[test]
    fn missing_part_errors() {
        let dir = tempfile::tempdir().unwrap();
        let parts = fixture_day(dir.path(), 2049);
        std::fs::remove_file(&parts[1]).unwrap();
        let e = MultiPart::open(&parts).err().unwrap().to_string();
        assert!(e.contains("tar.ab"), "{e}");
        // A partial download marker means the day is incomplete.
        std::fs::write(
            dir.path()
                .join("v2026.09.08-planes-readsb-prod-0.tar.ab.part"),
            b"x",
        )
        .unwrap();
        assert_eq!(local_parts(dir.path()), None);
    }
}
