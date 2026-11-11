//! What geomedia keeps: source data in the cache (ADS-B days, terrain tiles, geoid, weather,
//! aircraft types, geocoding replies) and kept data (media, runs) — from file names, sizes,
//! manifests, indexes and usage stamps (`aircraft_near` alone reads traces).

use std::path::{Path, PathBuf};

use crate::area::Area;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum DayState {
    /// Every part listed in the day's manifest is present at its expected size.
    Complete,
    /// No manifest (downloaded before manifests existed) and no partial download.
    Legacy,
    /// A partial download or a manifest that disagrees with the files.
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CachedDay {
    pub date: String,
    pub bytes: u64,
    pub state: DayState,
    pub indexed: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CachedTile {
    pub lat: i32,
    pub lon: i32,
    /// No GLO-30 tile exists there (open water): ground = geoid.
    pub ocean: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct CachedWeather {
    pub lat: f64,
    pub lon: f64,
    pub date: String,
}

/// One thing geomedia keeps, in a form `geomedia data` lists and `geomedia data clean` selects.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Item {
    /// adsb, terrain, geoid, weather, types, geocode, media, runs or partial.
    pub kind: String,
    /// The item's name within its kind: a date, a tile (N40W074), a file, a run id.
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    /// UNIX seconds: the usage stamp, else the newest modification time.
    pub last_used: Option<i64>,
    pub detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct Inventory {
    pub adsb_days: Vec<CachedDay>,
    pub tiles: Vec<CachedTile>,
    pub geoid_bytes: Option<u64>,
    pub weather: Vec<CachedWeather>,
    pub types_cached: bool,
    /// Everything above and the rest (geocoding, media, runs, partial downloads), uniformly.
    pub items: Vec<Item>,
}

/// Size of a file or directory tree (symlinks counted as themselves, never followed).
pub fn tree_bytes(p: &Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(p) else {
        return 0;
    };
    if !m.is_dir() {
        return m.len();
    }
    std::fs::read_dir(p)
        .map(|rd| rd.flatten().map(|e| tree_bytes(&e.path())).sum())
        .unwrap_or(0)
}

/// Newest modification time in a tree, UNIX seconds.
fn tree_mtime(p: &Path) -> Option<i64> {
    let m = std::fs::symlink_metadata(p).ok()?;
    let own = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);
    if !m.is_dir() {
        return own;
    }
    std::fs::read_dir(p)
        .ok()?
        .flatten()
        .filter_map(|e| tree_mtime(&e.path()))
        .chain(own)
        .max()
}

/// Every `*.part` (interrupted download) in the folders geomedia downloads into — never
/// elsewhere, in case the cache override names a folder shared with other programs.
fn partials(cache: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = [
        "dem/glo30",
        "geoid",
        "types",
        "geocode",
        "weather/openmeteo",
    ]
    .iter()
    .map(|d| cache.join(d))
    .collect();
    let adsb = cache.join("adsb");
    dirs.extend(
        subdirs(&adsb)
            .into_iter()
            .filter(|d| is_date(d))
            .map(|d| adsb.join(d)),
    );
    let mut out = Vec::new();
    for dir in dirs {
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let p = e.path();
            if e.file_type().is_ok_and(|t| t.is_file())
                && p.extension().is_some_and(|x| x == "part")
            {
                out.push(p);
            }
        }
    }
    out
}

fn tile_key(lat: i32, lon: i32) -> String {
    let ns = if lat < 0 { 'S' } else { 'N' };
    let ew = if lon < 0 { 'W' } else { 'E' };
    format!("{ns}{:02}{ew}{:03}", lat.unsigned_abs(), lon.unsigned_abs())
}

fn files(dir: &Path) -> Vec<(String, u64)> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<(String, u64)> = rd
        .flatten()
        .filter_map(|e| {
            // The entry itself, not a link's target: a symlink is listed (and cleaned) as the
            // link, sized as the link.
            let m = std::fs::symlink_metadata(e.path()).ok()?;
            (m.is_file() || m.file_type().is_symlink())
                .then(|| (e.file_name().to_string_lossy().into_owned(), m.len()))
        })
        .collect();
    v.sort();
    v
}

fn subdirs(dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<String> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && b.iter()
            .enumerate()
            .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit())
}

/// `Copernicus_DSM_COG_10_N40_00_W074_00_DEM` → (40, −74).
fn tile_latlon(name: &str) -> Option<(i32, i32)> {
    let rest = name
        .strip_prefix("Copernicus_DSM_COG_10_")?
        .strip_suffix("_DEM")?;
    let mut it = rest.split('_');
    let (la, _, lo, _) = (it.next()?, it.next()?, it.next()?, it.next()?);
    if it.next().is_some() {
        return None;
    }
    let sign = |c: char, pos: char, neg: char| match c {
        x if x == pos => Some(1),
        x if x == neg => Some(-1),
        _ => None,
    };
    let lat = sign(la.chars().next()?, 'N', 'S')? * la.get(1..)?.parse::<i32>().ok()?;
    let lon = sign(lo.chars().next()?, 'E', 'W')? * lo.get(1..)?.parse::<i32>().ok()?;
    Some((lat, lon))
}

pub fn inventory(cache: &Path, data: &Path) -> Inventory {
    let mut inv = Inventory::default();
    let used_c = crate::paths::last_used(cache);
    let used_d = crate::paths::last_used(data);
    let stamp =
        |used: &std::collections::HashMap<(String, String), i64>, k: &str, n: &str, p: &Path| {
            used.get(&(k.to_string(), n.to_string()))
                .copied()
                .or_else(|| tree_mtime(p))
        };
    let mut items: Vec<Item> = Vec::new();
    let adsb = cache.join("adsb");
    for d in subdirs(&adsb).into_iter().filter(|d| is_date(d)) {
        let dir = adsb.join(&d);
        let fs = files(&dir);
        if fs.is_empty() {
            continue;
        }
        let has_manifest = fs.iter().any(|(n, _)| n == "manifest.json");
        let complete = crate::adsblol::local_parts(&dir).is_some();
        let state = match (complete, has_manifest) {
            (true, true) => DayState::Complete,
            (true, false) => DayState::Legacy,
            (false, _) => DayState::Incomplete,
        };
        let day = CachedDay {
            bytes: fs.iter().map(|(_, n)| n).sum(),
            indexed: fs.iter().any(|(n, _)| n == "index.json"),
            date: d,
            state,
        };
        items.push(Item {
            kind: "adsb".into(),
            name: day.date.clone(),
            bytes: day.bytes,
            last_used: stamp(&used_c, "adsb", &day.date, &dir),
            detail: format!(
                "{}, {}",
                match day.state {
                    DayState::Complete => "complete",
                    DayState::Legacy => "complete (legacy)",
                    DayState::Incomplete => "INCOMPLETE",
                },
                if day.indexed {
                    "indexed"
                } else {
                    "not indexed"
                }
            ),
            path: dir.clone(),
        });
        inv.adsb_days.push(day);
    }
    let dem = cache.join("dem/glo30");
    for (n, bytes) in files(&dem) {
        let (stem, ocean) = match (n.strip_suffix(".tif"), n.strip_suffix(".ocean")) {
            (Some(s), _) => (s, false),
            (_, Some(s)) => (s, true),
            _ => continue,
        };
        if let Some((lat, lon)) = tile_latlon(stem) {
            inv.tiles.push(CachedTile { lat, lon, ocean });
            let path = dem.join(&n);
            items.push(Item {
                kind: "terrain".into(),
                name: tile_key(lat, lon),
                last_used: stamp(&used_c, "terrain", stem, &path),
                path,
                bytes,
                detail: if ocean {
                    "open water (no tile)".into()
                } else {
                    "GLO-30 tile".into()
                },
            });
        }
    }
    inv.tiles.sort_by_key(|t| (t.lat, t.lon, t.ocean));
    inv.tiles.dedup();
    let geoid: u64 = files(&cache.join("geoid")).iter().map(|(_, n)| n).sum();
    inv.geoid_bytes = (geoid > 0).then_some(geoid);
    // Flat folders of files: one item per file.
    for (kind, sub, detail) in [
        ("geoid", "geoid", "EGM2008 geoid"),
        ("types", "types", "aircraft type table"),
        ("geocode", "geocode", "place-name lookup"),
        ("weather", "weather/openmeteo", "Open-Meteo profile"),
    ] {
        let dir = cache.join(sub);
        for (n, bytes) in files(&dir) {
            if n.ends_with(".part") {
                continue;
            }
            let path = dir.join(&n);
            items.push(Item {
                kind: kind.into(),
                last_used: stamp(&used_c, kind, &n, &path),
                name: n,
                path,
                bytes,
                detail: detail.into(),
            });
        }
    }
    for (n, _) in files(&cache.join("weather/openmeteo")) {
        let Some(stem) = n.strip_suffix(".json") else {
            continue;
        };
        let mut it = stem.splitn(3, '_');
        if let (Some(la), Some(lo), Some(date)) = (it.next(), it.next(), it.next())
            && let (Ok(lat), Ok(lon)) = (la.parse::<f64>(), lo.parse::<f64>())
            && is_date(date)
        {
            inv.weather.push(CachedWeather {
                lat,
                lon,
                date: date.to_string(),
            });
        }
    }
    inv.types_cached = !files(&cache.join("types")).is_empty();
    // Recordings: the kept folder, plus any a failed migration left in the cache — listed so
    // they're never mistaken for re-downloadable source data.
    for (media, detail) in [
        (data.join("media"), "recording"),
        (
            cache.join("media"),
            "recording, not migrated (still in the source-data folder)",
        ),
    ] {
        for e in std::fs::read_dir(&media).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let path = e.path();
            items.push(Item {
                kind: "media".into(),
                bytes: tree_bytes(&path),
                last_used: stamp(&used_d, "media", &name, &path),
                detail: detail.into(),
                name,
                path,
            });
        }
    }
    let runs = data.join("runs");
    let mut run_items: Vec<Item> = Vec::new();
    for e in std::fs::read_dir(&runs).into_iter().flatten().flatten() {
        let path = e.path();
        if !path.is_dir() {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        let info: serde_json::Value = std::fs::read(path.join("run.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let field = |k: &str| info[k].as_str().unwrap_or("").to_string();
        let file = field("file");
        let file = Path::new(&file)
            .file_name()
            .map_or(file.clone(), |f| f.to_string_lossy().into_owned());
        run_items.push(Item {
            kind: "runs".into(),
            bytes: tree_bytes(&path),
            last_used: tree_mtime(&path),
            detail: [field("kind"), field("verdict"), file]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" · "),
            name,
            path,
        });
    }
    run_items.sort_by(|a, b| a.name.cmp(&b.name));
    items.extend(run_items);
    let mut parts = partials(cache);
    parts.sort();
    for path in parts {
        items.push(Item {
            kind: "partial".into(),
            name: path
                .strip_prefix(cache)
                .unwrap_or(&path)
                .display()
                .to_string(),
            bytes: tree_bytes(&path),
            last_used: tree_mtime(&path),
            detail: "interrupted download".into(),
            path,
        });
    }
    inv.items = items;
    inv
}

/// Aircraft that come within `query::PAD_M` of `area` at some point on `date` (UTC day): the
/// day index picks candidates, then only those traces are read (tens of MB, not the archive).
/// `None` when the day isn't complete and indexed.
pub fn aircraft_near(cache: &Path, date: &str, area: &Area) -> Option<usize> {
    let dir = cache.join("adsb").join(date);
    let text = std::fs::read(dir.join("index.json")).ok()?;
    let idx: crate::adsblol::DayIndex = serde_json::from_slice(&text).ok()?;
    let parts = crate::adsblol::local_parts(&dir)?;
    let reader = crate::adsblol::MultiPart::open(&parts).ok()?;
    let t0 = gm_core::Epoch::from_rfc3339(&format!("{date}T00:00:00Z"))?;
    let t1 = t0.plus_seconds(86_400.0);
    crate::query::query_reader(&idx, reader, area, t0, t1, t0)
        .ok()
        .map(|v| v.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adsblol::tests::fixture_day;
    use crate::adsblol::{index_parts, write_manifest};

    fn date() -> time::Date {
        time::Date::from_calendar_date(2026, time::Month::September, 8).unwrap()
    }

    /// A cache with one complete indexed day, 2 land tiles + 1 ocean marker, the geoid, a
    /// weather site and the type table.
    fn cache() -> tempfile::TempDir {
        let c = tempfile::tempdir().unwrap();
        let day = c.path().join("adsb/2026-09-08");
        std::fs::create_dir_all(&day).unwrap();
        let parts = fixture_day(&day, 2049);
        let sizes: Vec<(String, u64)> = parts
            .iter()
            .map(|p| {
                let n = p.file_name().unwrap().to_string_lossy().into_owned();
                (n, std::fs::metadata(p).unwrap().len())
            })
            .collect();
        write_manifest(&day, &sizes).unwrap();
        index_parts(date(), &parts, &day).unwrap();
        let dem = c.path().join("dem/glo30");
        std::fs::create_dir_all(&dem).unwrap();
        for n in [
            "Copernicus_DSM_COG_10_N40_00_W074_00_DEM.tif",
            "Copernicus_DSM_COG_10_S01_00_E028_00_DEM.tif",
            "Copernicus_DSM_COG_10_N40_00_W073_00_DEM.ocean",
        ] {
            std::fs::write(dem.join(n), b"x").unwrap();
        }
        let geo = c.path().join("geoid");
        std::fs::create_dir_all(&geo).unwrap();
        std::fs::write(geo.join("us_nga_egm08_25.tif"), vec![0u8; 100]).unwrap();
        let wx = c.path().join("weather/openmeteo");
        std::fs::create_dir_all(&wx).unwrap();
        std::fs::write(wx.join("40.88_-73.51_2026-09-08.json"), b"{}").unwrap();
        std::fs::create_dir_all(c.path().join("types")).unwrap();
        std::fs::write(c.path().join("types/types.json"), b"{}").unwrap();
        c
    }

    #[test]
    fn inventory_lists_runs_media_partials() {
        let c = cache();
        let d = tempfile::tempdir().unwrap();
        let run = d.path().join("runs/2026-11-04T16-32-10_locate_a");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(
            run.join("run.json"),
            r#"{"kind":"locate","file":"/x/a.wav","verdict":"located","created":"2026-11-04T16:32:10Z"}"#,
        )
        .unwrap();
        std::fs::write(run.join("locate.json"), b"{}").unwrap();
        std::fs::create_dir_all(d.path().join("media/oysterbay")).unwrap();
        std::fs::write(d.path().join("media/oysterbay/raw.webm"), b"12345").unwrap();
        std::fs::write(c.path().join("adsb/2026-09-08/v.tar.ac.part"), b"xx").unwrap();
        crate::paths::touch_used_in(c.path(), "adsb", "2026-09-08");
        let inv = inventory(c.path(), d.path());
        let item = |kind: &str, name: &str| {
            inv.items
                .iter()
                .find(|i| i.kind == kind && i.name == name)
                .unwrap_or_else(|| panic!("{kind} {name} missing: {:#?}", inv.items))
                .clone()
        };
        let r = item("runs", "2026-11-04T16-32-10_locate_a");
        assert!(
            r.detail.contains("located") && r.detail.contains("a.wav"),
            "{r:?}"
        );
        assert!(r.bytes > 0);
        assert_eq!(item("media", "oysterbay").bytes, 5);
        let day = item("adsb", "2026-09-08");
        let now = time::OffsetDateTime::now_utc().unix_timestamp();
        assert!(
            day.last_used.is_some_and(|t| (now - t).abs() < 5),
            "{day:?}"
        );
        let part = inv
            .items
            .iter()
            .find(|i| i.kind == "partial")
            .expect("partial listed");
        assert!(
            part.path.ends_with("adsb/2026-09-08/v.tar.ac.part"),
            "{part:?}"
        );
        assert_eq!(item("terrain", "N40W074").kind, "terrain");
        assert!(inv.items.iter().all(|i| i.path.is_absolute()));
    }

    #[test]
    fn inventory_lists_unmigrated_media() {
        let c = cache();
        std::fs::create_dir_all(c.path().join("media/left")).unwrap();
        std::fs::write(c.path().join("media/left/raw.webm"), b"123").unwrap();
        let inv = inventory(c.path(), Path::new("/nonexistent/geomedia-data"));
        let m = inv
            .items
            .iter()
            .find(|i| i.kind == "media" && i.name == "left")
            .unwrap_or_else(|| panic!("{:#?}", inv.items));
        assert!(m.detail.contains("not migrated"), "{m:?}");
    }

    #[test]
    fn partials_only_in_geomedia_folders() {
        let c = cache();
        for p in ["Downloads/movie.mkv.part", "Documents/proj/sub/build.part"] {
            let p = c.path().join(p);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }
        std::fs::write(c.path().join("adsb/2026-09-08/v.tar.ab.part"), b"x").unwrap();
        std::fs::write(c.path().join("geoid/g.tif.part"), b"x").unwrap();
        let inv = inventory(c.path(), Path::new("/nonexistent/geomedia-data"));
        let parts: Vec<_> = inv.items.iter().filter(|i| i.kind == "partial").collect();
        assert_eq!(parts.len(), 2, "{parts:#?}");
        assert!(
            parts
                .iter()
                .all(|p| !p.name.contains("Downloads") && !p.name.contains("Documents"))
        );
    }

    #[test]
    fn inventory_lists_everything() {
        let c = cache();
        let inv = inventory(c.path(), Path::new("/nonexistent/geomedia-data"));
        assert_eq!(inv.adsb_days.len(), 1, "{inv:?}");
        let d = &inv.adsb_days[0];
        assert_eq!(d.date, "2026-09-08");
        assert_eq!(d.state, DayState::Complete);
        assert!(d.indexed && d.bytes > 0);
        assert_eq!(
            inv.tiles,
            vec![
                CachedTile {
                    lat: -1,
                    lon: 28,
                    ocean: false
                },
                CachedTile {
                    lat: 40,
                    lon: -74,
                    ocean: false
                },
                CachedTile {
                    lat: 40,
                    lon: -73,
                    ocean: true
                },
            ]
        );
        assert_eq!(inv.geoid_bytes, Some(100));
        assert_eq!(
            inv.weather,
            vec![CachedWeather {
                lat: 40.88,
                lon: -73.51,
                date: "2026-09-08".into()
            }]
        );
        assert!(inv.types_cached);
    }

    #[test]
    fn inventory_flags_incomplete_day() {
        let c = cache();
        let day = c.path().join("adsb/2026-09-08");
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(day.join("manifest.json")).unwrap()).unwrap();
        let mut m = m;
        m[0]["size"] = serde_json::json!(m[0]["size"].as_u64().unwrap() + 1);
        std::fs::write(day.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        assert_eq!(
            inventory(c.path(), Path::new("/nonexistent/geomedia-data")).adsb_days[0].state,
            DayState::Incomplete
        );
        // Legacy (no manifest) with a partial download is incomplete too.
        std::fs::remove_file(day.join("manifest.json")).unwrap();
        assert_eq!(
            inventory(c.path(), Path::new("/nonexistent/geomedia-data")).adsb_days[0].state,
            DayState::Legacy
        );
        std::fs::write(
            day.join("v2026.09.08-planes-readsb-prod-0.tar.ac.part"),
            b"x",
        )
        .unwrap();
        assert_eq!(
            inventory(c.path(), Path::new("/nonexistent/geomedia-data")).adsb_days[0].state,
            DayState::Incomplete
        );
    }

    #[test]
    fn inventory_empty_cache() {
        let inv = inventory(
            Path::new("/nonexistent/geomedia-cache"),
            Path::new("/nonexistent/geomedia-data"),
        );
        assert_eq!(inv, Inventory::default());
    }

    #[test]
    fn inventory_ignores_strays() {
        let c = cache();
        std::fs::write(c.path().join("adsb/.DS_Store"), b"x").unwrap();
        std::fs::create_dir_all(c.path().join("adsb/notes")).unwrap();
        std::fs::write(c.path().join("dem/glo30/foo.tif"), b"x").unwrap();
        std::fs::write(c.path().join("weather/openmeteo/readme.txt"), b"x").unwrap();
        let inv = inventory(c.path(), Path::new("/nonexistent/geomedia-data"));
        assert_eq!(inv.adsb_days.len(), 1);
        assert_eq!(inv.tiles.len(), 3);
        assert_eq!(inv.weather.len(), 1);
    }

    #[test]
    fn aircraft_near_counts_actual_passes() {
        // An L-shaped flight: its day bounding box covers (41.9, −73.9), but its path never comes
        // within 100 km of it. Counting from the index alone would say 1; the truth is 0.
        let c = tempfile::tempdir().unwrap();
        let day = c.path().join("adsb/2026-09-08");
        std::fs::create_dir_all(&day).unwrap();
        let mut b = tar::Builder::new(Vec::new());
        let mut add = |name: &str, data: &[u8]| {
            let mut h = tar::Header::new_gnu();
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            h.set_cksum();
            b.append_data(&mut h, name, data).unwrap();
        };
        use crate::adsblol::tests::{gz, trace_json};
        add(
            "./traces/01/trace_full_ab0001.json",
            &gz(&trace_json(
                "ab0001",
                1_788_825_600.0,
                &[
                    (75000.0, 40.0, -74.0),
                    (75600.0, 40.0, -72.0),
                    (76200.0, 42.0, -72.0),
                ],
            )),
        );
        let part = day.join("v2026.09.08-planes-readsb-prod-0.tar");
        std::fs::write(&part, b.into_inner().unwrap()).unwrap();
        index_parts(date(), std::slice::from_ref(&part), &day).unwrap();
        let corner = Area::Circle {
            lat: 41.9,
            lon: -73.9,
            radius_m: 5000.0,
        };
        let on_path = Area::Circle {
            lat: 40.0,
            lon: -73.0,
            radius_m: 5000.0,
        };
        assert_eq!(aircraft_near(c.path(), "2026-09-08", &corner), Some(0));
        assert_eq!(aircraft_near(c.path(), "2026-09-08", &on_path), Some(1));
        std::fs::remove_file(day.join("index.json")).unwrap();
        assert_eq!(aircraft_near(c.path(), "2026-09-08", &on_path), None);
    }
}
