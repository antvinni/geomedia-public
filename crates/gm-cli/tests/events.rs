mod common;

use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn sim_wav(dir: &Path) -> PathBuf {
    let out = common::geomedia()
        .arg("sim")
        .arg(fixture("sim/two_props.toml"))
        .arg("--out")
        .arg(dir)
        .output()
        .unwrap();
    assert!(out.status.success());
    dir.join("audio.wav")
}

fn events(args: &[&str], file: &Path) -> (i32, String, String) {
    let o = common::geomedia()
        .arg("events")
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
fn events_cli() {
    let dir = tempfile::tempdir().unwrap();
    let wav = sim_wav(dir.path());

    let (code, out, err) = events(&[], &wav);
    assert_eq!(code, 0, "{err}");
    // The aircraft-sound log comes first (two quiet tonal props: no loudness hump), then the
    // Doppler events.
    assert!(
        out.lines()
            .next()
            .is_some_and(|l| l == "no aircraft sounds found" || l.starts_with("pass ")),
        "{out}"
    );
    let lines: Vec<&str> = out.lines().filter(|l| l.starts_with('#')).collect();
    assert_eq!(lines.len(), 2, "{out}");
    assert!(
        lines[0].starts_with("#1  tc ") && lines[1].starts_with("#2  tc "),
        "{out}"
    );
    assert!(
        lines[0].contains(" m/s") && lines[0].contains(" Hz (k="),
        "{out}"
    );

    let (code, out, _) = events(&["--event", "15:00-16:40"], &wav);
    assert_eq!(code, 0);
    assert_eq!(
        out.lines().filter(|l| l.starts_with('#')).count(),
        1,
        "{out}"
    );

    let (code, _, err) = events(&["--event", "40-10"], &wav);
    assert_eq!(code, 1);
    assert!(err.contains("40-10"), "{err}");
    let (code, _, err) = events(&["--event", "5000-5100"], &wav);
    assert_eq!(code, 1);
    assert!(err.contains("5000-5100"), "{err}");

    let (code, out, _) = events(&["--json", "-"], &wav);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["events"].as_array().unwrap().len(), 2);
    assert_eq!(v["schema_version"], 2);
    assert!(v["passes"].is_array() && v["rejected"].is_array(), "{out}");
}

#[test]
fn events_none_found() {
    let (code, out, err) = events(&[], &fixture("plain.wav"));
    assert_eq!(code, 0, "{err}");
    // A 1-s file: nothing to judge, logged with its reason, and no events.
    assert!(
        out.contains("too short to judge (recording 1.0 s)"),
        "{out}"
    );
    assert_eq!(out.lines().last(), Some("no Doppler events found"), "{out}");
}
