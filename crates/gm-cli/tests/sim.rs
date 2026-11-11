mod common;

use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn sim_writes_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let out = common::geomedia()
        .args(["sim"])
        .arg(fixture("sim/two_props.toml"))
        .arg("--out")
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.starts_with("wrote "), "{stdout}");
    let meta = common::geomedia()
        .arg("meta")
        .arg(dir.path().join("audio.wav"))
        .arg("--no-report")
        .output()
        .unwrap();
    let m = String::from_utf8(meta.stdout).unwrap();
    assert!(m.contains("  wav") && m.contains("8000 Hz"), "{m}");
    let truth: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("truth.json")).unwrap()).unwrap();
    let ev = truth["events"].as_array().unwrap();
    assert_eq!(ev.len(), 2);
    let dt = ev[1]["cpa_arrival_audio_s"].as_f64().unwrap()
        - ev[0]["cpa_arrival_audio_s"].as_f64().unwrap();
    assert!((dt - 920.0).abs() < 2.0, "{dt}");
}

#[test]
fn sim_bad_scenario_exit_1() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.toml");
    std::fs::write(&bad, "seed = 1\n").unwrap();
    let out = common::geomedia()
        .arg("sim")
        .arg(&bad)
        .arg("--out")
        .arg(dir.path())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).starts_with("error:"));
}
