mod common;

use std::path::Path;

fn data(cache: &Path, kept: &Path, args: &[&str]) -> (bool, String, String) {
    let out = common::geomedia()
        .arg("data")
        .args(args)
        .env("GEOMEDIA_CACHE", cache)
        .env("GEOMEDIA_DATA", kept)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn populate(cache: &Path, kept: &Path) {
    let day = cache.join("adsb/2026-09-08");
    std::fs::create_dir_all(&day).unwrap();
    std::fs::write(day.join("v2026.09.08-planes-readsb-prod-0.tar.aa"), b"xx").unwrap();
    let dem = cache.join("dem/glo30");
    std::fs::create_dir_all(&dem).unwrap();
    std::fs::write(
        dem.join("Copernicus_DSM_COG_10_N40_00_W074_00_DEM.tif"),
        b"x",
    )
    .unwrap();
    let run = kept.join("runs/2026-11-04T16-32-10_locate_a");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(
        run.join("run.json"),
        r#"{"kind":"locate","file":"/x/a.wav","verdict":"located"}"#,
    )
    .unwrap();
    std::fs::create_dir_all(kept.join("media/oysterbay")).unwrap();
    std::fs::write(kept.join("media/oysterbay/raw.webm"), b"12345").unwrap();
}

#[test]
fn data_lists_everything() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let (ok, out, err) = data(c.path(), k.path(), &[]);
    assert!(ok, "{out}{err}");
    assert!(
        out.contains(&c.path().display().to_string()),
        "cache path: {out}"
    );
    assert!(
        out.contains(&k.path().display().to_string()),
        "data path: {out}"
    );
    for want in [
        "ADS-B days",
        "2026-09-08",
        "terrain",
        "N40W074",
        "media",
        "oysterbay",
        "runs",
        "2026-11-04T16-32-10_locate_a",
        "located",
    ] {
        assert!(out.contains(want), "{want} missing:\n{out}");
    }
}

#[test]
fn data_area_and_json() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let (ok, out, err) = data(c.path(), k.path(), &["--area", "40.80,-73.50~30km"]);
    assert!(ok, "{out}{err}");
    assert!(
        out.contains("not indexed") && out.contains("terrain for area"),
        "{out}"
    );
    assert!(err.contains("checking 2026-09-08"), "progress line: {err}");
    let (ok, j, _) = data(c.path(), k.path(), &["--json"]);
    assert!(ok, "{j}");
    let v: serde_json::Value = serde_json::from_str(&j).unwrap();
    assert_eq!(v["adsb_days"][0]["date"], "2026-09-08");
    assert!(
        v["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| i["kind"] == "media")
    );
}

#[test]
fn data_fresh_machine() {
    let base = tempfile::tempdir().unwrap();
    let (c, k) = (
        base.path().join("nope-cache"),
        base.path().join("nope-data"),
    );
    let (ok, out, err) = data(&c, &k, &[]);
    assert!(ok, "{out}{err}");
    assert!(out.contains("nothing kept"), "{out}");
    assert!(out.contains(&c.display().to_string()), "{out}");
}

fn clean(cache: &Path, kept: &Path, args: &[&str]) -> (bool, String, String) {
    let mut a = vec!["clean"];
    a.extend_from_slice(args);
    data(cache, kept, &a)
}

/// Set a file's modification time `days` ago.
fn age(p: &Path, days: u64) {
    let t = std::time::SystemTime::now() - std::time::Duration::from_secs(days * 86_400);
    std::fs::File::options()
        .write(true)
        .open(p)
        .unwrap()
        .set_modified(t)
        .unwrap();
}

#[test]
fn clean_adsb_day() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let other = c.path().join("adsb/2026-09-05");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(
        other.join("v2026.09.05-planes-readsb-prod-0.tar.aa"),
        b"yyy",
    )
    .unwrap();
    let (ok, out, err) = clean(c.path(), k.path(), &["adsb", "2026-09-08", "--yes"]);
    assert!(ok, "{out}{err}");
    assert!(!c.path().join("adsb/2026-09-08").exists(), "{out}");
    assert!(other.is_dir());
    assert!(
        c.path()
            .join("dem/glo30/Copernicus_DSM_COG_10_N40_00_W074_00_DEM.tif")
            .is_file()
    );
    assert!(k.path().join("media/oysterbay/raw.webm").is_file());
    assert!(out.contains("freed 2 B"), "{out}");
}

#[test]
fn clean_unused_for() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let dem = c.path().join("dem/glo30");
    let old = dem.join("Copernicus_DSM_COG_10_N40_00_W074_00_DEM.tif");
    let new = dem.join("Copernicus_DSM_COG_10_N41_00_W074_00_DEM.tif");
    std::fs::write(&new, b"x").unwrap();
    age(&old, 200);
    let (ok, out, err) = clean(
        c.path(),
        k.path(),
        &["terrain", "--unused-for", "90d", "--yes"],
    );
    assert!(ok, "{out}{err}");
    assert!(!old.exists(), "{out}");
    assert!(new.is_file(), "{out}");
}

#[test]
fn clean_never_follows_links_out() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let outside = tempfile::tempdir().unwrap();
    populate(c.path(), k.path());
    let target = outside.path().join("precious.tif");
    std::fs::write(&target, b"keep me").unwrap();
    let tdir = outside.path().join("precious-dir");
    std::fs::create_dir_all(&tdir).unwrap();
    std::fs::write(tdir.join("inner.webm"), b"keep me too").unwrap();
    std::fs::create_dir_all(c.path().join("geoid")).unwrap();
    std::os::unix::fs::symlink(&target, c.path().join("geoid/egm.tif")).unwrap();
    std::os::unix::fs::symlink(&tdir, k.path().join("media/linked")).unwrap();
    let (ok, out, err) = clean(c.path(), k.path(), &["all", "--yes"]);
    assert!(ok, "{out}{err}");
    assert!(out.contains("geoid egm.tif"), "link listed: {out}");
    assert!(target.is_file() && tdir.join("inner.webm").is_file());
    assert!(std::fs::symlink_metadata(c.path().join("geoid/egm.tif")).is_err());
    assert!(std::fs::symlink_metadata(k.path().join("media/linked")).is_err());
    assert!(!c.path().join("adsb/2026-09-08").exists());
}

#[test]
fn clean_refuses_without_yes_non_interactive() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let (ok, out, err) = clean(c.path(), k.path(), &["all"]);
    assert!(ok, "{out}{err}");
    assert!(
        out.contains("2026-09-08") && out.contains("oysterbay"),
        "{out}"
    );
    assert!(out.contains("recordings"), "media warning: {out}");
    assert!(out.contains("nothing deleted (add --yes)"), "{out}");
    assert!(c.path().join("adsb/2026-09-08").is_dir());
    assert!(k.path().join("media/oysterbay/raw.webm").is_file());
}

#[test]
fn clean_partial_only() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let day = c.path().join("adsb/2026-09-08");
    let part = day.join("v2026.09.08-planes-readsb-prod-0.tar.ab.part");
    std::fs::write(&part, b"zz").unwrap();
    let (ok, out, err) = clean(c.path(), k.path(), &["partial", "--yes"]);
    assert!(ok, "{out}{err}");
    assert!(!part.exists(), "{out}");
    assert!(
        day.join("v2026.09.08-planes-readsb-prod-0.tar.aa")
            .is_file()
    );
}

#[test]
fn clean_runs_by_id() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let other = k.path().join("runs/2026-11-04T17-00-00_meta_b");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("run.json"), b"{}").unwrap();
    let (ok, out, err) = clean(
        c.path(),
        k.path(),
        &["runs", "2026-11-04T16-32-10_locate_a", "--yes"],
    );
    assert!(ok, "{out}{err}");
    assert!(!k.path().join("runs/2026-11-04T16-32-10_locate_a").exists());
    assert!(other.is_dir());
    // An unknown item is an error, not a silent no-op.
    let (ok, out, err) = clean(c.path(), k.path(), &["runs", "no-such-run", "--yes"]);
    assert!(!ok, "{out}{err}");
    assert!(err.contains("no-such-run"), "{err}");
}

#[test]
fn clean_skips_downloading_day_unless_named() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let day = c.path().join("adsb/2026-09-08");
    std::fs::write(
        day.join("v2026.09.08-planes-readsb-prod-0.tar.ab.part"),
        b"zz",
    )
    .unwrap();
    for what in ["all", "adsb"] {
        let (ok, out, err) = clean(c.path(), k.path(), &[what, "--yes"]);
        assert!(ok, "{out}{err}");
        assert!(day.is_dir(), "{what} removed a day being downloaded: {out}");
        assert!(
            out.contains("skipped") && out.contains("2026-09-08"),
            "{out}"
        );
    }
    let (ok, out, err) = clean(c.path(), k.path(), &["adsb", "2026-09-08", "--yes"]);
    assert!(ok, "{out}{err}");
    assert!(!day.exists(), "{out}");
}

#[test]
fn clean_refuses_overlapping_folders() {
    let (c, k) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    populate(c.path(), k.path());
    let (ok, out, err) = clean(c.path(), c.path(), &["all", "--yes"]);
    assert!(!ok, "{out}{err}");
    assert!(err.contains("refusing"), "{err}");
    assert!(c.path().join("adsb/2026-09-08").is_dir());
}
