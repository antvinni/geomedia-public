mod common;

use flate2::Compression;
use flate2::write::GzEncoder;
use std::io::Write;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn gz(s: &str) -> Vec<u8> {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(s.as_bytes()).unwrap();
    e.finish().unwrap()
}

/// A cache with a tiny 2026-09-08 ADS-B day (two aircraft near Oyster Bay), terrain and geoid.
fn cache() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let day = dir.path().join("adsb/2026-09-08");
    std::fs::create_dir_all(&day).unwrap();
    let trace = |icao: &str, t0: f64, lat: f64, lon: f64| {
        let pts: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    "[{},{},{},5000,120.0,90.0,0,0,null,\"adsb_icao\",5200,null,null,null]",
                    t0 + 10.0 * f64::from(i),
                    lat,
                    lon + 0.002 * f64::from(i)
                )
            })
            .collect();
        gz(&format!(
            r#"{{"icao":"{icao}","r":"N1","t":"PC12","timestamp":1788825600.0,"trace":[{}]}}"#,
            pts.join(",")
        ))
    };
    let mut b = tar::Builder::new(Vec::new());
    for (name, data) in [
        (
            "./traces/eb/trace_full_a1dfeb.json",
            trace("a1dfeb", 75_000.0, 40.85, -73.60),
        ),
        (
            "./traces/91/trace_full_ace391.json",
            trace("ace391", 74_900.0, 40.90, -73.50),
        ),
        (
            "./traces/02/trace_full_aa0002.json",
            trace("aa0002", 75_000.0, 51.47, -0.45),
        ),
    ] {
        let mut h = tar::Header::new_gnu();
        h.set_size(data.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        b.append_data(&mut h, name, &data[..]).unwrap();
    }
    std::fs::write(
        day.join("v2026.09.08-planes-readsb-prod-0.tar"),
        b.into_inner().unwrap(),
    )
    .unwrap();
    let dem = dir.path().join("dem/glo30");
    std::fs::create_dir_all(&dem).unwrap();
    std::fs::write(
        dem.join("Copernicus_DSM_COG_10_N40_00_W074_00_DEM.ocean"),
        b"",
    )
    .unwrap();
    std::fs::create_dir_all(dir.path().join("geoid")).unwrap();
    std::fs::write(dir.path().join("geoid/us_nga_egm08_25.tif"), b"").unwrap();
    dir
}

fn preflight(cache: &Path, args: &[&str], file: &Path) -> (i32, String, String) {
    let o = common::geomedia()
        .env("GEOMEDIA_CACHE", cache)
        .arg("preflight")
        .arg(file)
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8(o.stdout).unwrap(),
        String::from_utf8(o.stderr).unwrap(),
    )
}

#[test]
fn preflight_offline_report() {
    let c = cache();
    let (code, out, err) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
        ],
        &fixture("plain.wav"),
    );
    assert_eq!(code, 0, "{err}\n{out}");
    assert!(out.contains("Area: circle"), "{out}");
    assert!(out.contains("2026-09-08: cached"), "{out}");
    assert!(out.contains("2 aircraft within 30 km"), "{out}");
    assert!(out.contains("terrain: 1 tile"), "{out}");
    assert!(out.trim_end().ends_with("ready"), "{out}");
}

#[test]
fn preflight_offers_metadata_never_applies() {
    let c = cache();
    let (code, out, err) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
        ],
        &fixture("phone_gps.mov"),
    );
    assert_eq!(code, 0, "{err}");
    assert!(
        out.contains("metadata suggests --area 51.5072,-0.1276"),
        "{out}"
    );
    assert!(out.contains("(untrusted"), "{out}");
    assert!(out.contains("40.876500"), "the user's area is used: {out}");
}

#[test]
fn preflight_no_zone_error() {
    let c = cache();
    let (code, _, err) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08 20:50",
        ],
        &fixture("plain.wav"),
    );
    assert_eq!(code, 1);
    assert!(err.contains("time zone"), "{err}");
}

#[test]
fn preflight_needs_downloads() {
    let c = cache();
    std::fs::remove_file(c.path().join("geoid/us_nga_egm08_25.tif")).unwrap();
    let (code, out, _) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
        ],
        &fixture("plain.wav"),
    );
    assert_eq!(code, 0);
    assert!(
        out.contains("needs downloads") && out.contains("--yes"),
        "{out}"
    );
}

#[test]
fn preflight_measures_duration_when_container_lacks_it() {
    if std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_err()
    {
        eprintln!("skipped: ffmpeg not installed");
        return;
    }
    let c = cache();
    let (code, out, err) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
        ],
        &fixture("opus.webm"),
    );
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("(3.0 s"), "{out}");
}

#[test]
fn preflight_reports_ocean_tiles() {
    let c = cache();
    let (_, out, _) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
        ],
        &fixture("plain.wav"),
    );
    assert!(out.contains("1 assumed sea level"), "{out}");
}

fn preflight_raw(cache: &Path, args: &[&str]) -> (i32, String, String) {
    let o = common::geomedia()
        .env("GEOMEDIA_CACHE", cache)
        .arg("preflight")
        .args(args)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8(o.stdout).unwrap(),
        String::from_utf8(o.stderr).unwrap(),
    )
}

#[test]
fn preflight_without_file() {
    // Fixture day cached (no network); geoid removed so a download is still needed.
    let c = cache();
    std::fs::remove_file(c.path().join("geoid/us_nga_egm08_25.tif")).unwrap();
    let (code, out, err) = preflight_raw(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~2km",
            "--when",
            "2026-09-08T20:17:29Z±5s",
            "--duration",
            "1h",
        ],
    );
    assert_eq!(code, 0, "{err}\n{out}");
    assert!(out.contains("metadata not checked"), "{out}");
    assert!(out.contains("3600 s"), "{out}");
    assert!(out.contains("2026-09-08: cached"), "{out}");
    assert!(out.contains("needs downloads: EGM2008"), "{out}");
}

#[test]
fn preflight_unknown_length_uses_duration() {
    let c = cache();
    let (code, out, err) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
            "--duration",
            "1h",
        ],
        &fixture("hello.bin"),
    );
    assert_eq!(code, 0, "{err}\n{out}");
    assert!(out.contains("(3600.0 s)"), "{out}");
    assert!(out.contains("length unknown; using --duration"), "{out}");
}

#[test]
fn preflight_future_day_not_published() {
    // A future recording time: adsb.lol hasn't published that day; no request, no --yes advice.
    let c = cache();
    let (code, out, err) = preflight_raw(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~2km",
            "--when",
            "2099-01-01T12:00:00Z±5s",
            "--duration",
            "30m",
        ],
    );
    assert_eq!(code, 0, "{err}\n{out}");
    assert!(out.contains("2099-01-01: not published yet"), "{out}");
    assert!(!out.contains("--yes"), "{out}");
}

#[test]
fn preflight_needs_file_or_duration() {
    let dir = tempfile::tempdir().unwrap();
    let (code, out, err) = preflight_raw(
        dir.path(),
        &[
            "--area",
            "40.8765,-73.525~2km",
            "--when",
            "2026-09-08T20:17:29Z±5s",
        ],
    );
    assert_ne!(code, 0, "{out}");
    assert!(err.contains("--duration"), "{err}");
}

#[test]
fn preflight_file_wins_over_duration() {
    let c = cache();
    let (code, out, err) = preflight(
        c.path(),
        &[
            "--area",
            "40.8765,-73.525~10km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
            "--duration",
            "1h",
        ],
        &fixture("plain.wav"),
    );
    assert_eq!(code, 0, "{err}\n{out}");
    assert!(out.contains("(1.0 s)"), "{out}");
    assert!(out.contains("--duration ignored"), "{out}");
}

#[test]
fn preflight_flags_no_adsb_coverage() {
    // Terrain cached, ADS-B day cached, but no aircraft within 30 km in the window: not "ready".
    let c = cache();
    let (code, out, err) = preflight(
        c.path(),
        &[
            "--area",
            "40.45,-73.95~2km",
            "--when",
            "2026-09-08T20:50:00Z±5s",
        ],
        &fixture("plain.wav"),
    );
    assert_eq!(code, 0, "{err}\n{out}");
    assert!(out.contains("0 aircraft within 30 km"), "{out}");
    assert!(out.contains("no ADS-B coverage"), "{out}");
    assert!(!out.trim_end().ends_with("ready"), "{out}");
}
