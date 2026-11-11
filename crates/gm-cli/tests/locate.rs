mod common;

use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn sim(dir: &Path) -> (PathBuf, PathBuf) {
    let out = common::geomedia()
        .arg("sim")
        .arg(fixture("sim/two_props.toml"))
        .arg("--out")
        .arg(dir)
        .output()
        .unwrap();
    assert!(out.status.success());
    (dir.join("audio.wav"), dir.join("truth.json"))
}

fn locate(cache: &Path, file: &Path, args: &[&str]) -> (i32, String, String) {
    let o = common::geomedia()
        .env("GEOMEDIA_CACHE", cache)
        .arg("locate")
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
fn locate_sim_two_props() {
    let dir = tempfile::tempdir().unwrap();
    let (wav, truth) = sim(dir.path());
    let out = dir.path().join("report");
    // Area centred ~1.4 km from the truth (40.8765, -73.5250).
    let (code, stdout, stderr) = locate(
        dir.path(),
        &wav,
        &[
            "--area",
            "40.8840,-73.5110~5km",
            "--when",
            "2026-09-08T20:40:00Z±5s",
            "--sim-truth",
            truth.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(stdout.contains("located"), "{stdout}");
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("locate.json")).unwrap()).unwrap();
    assert_eq!(j["schema_version"], 1);
    let (lat, lon) = (
        j["best"]["geo"]["lat_deg"].as_f64().unwrap(),
        j["best"]["geo"]["lon_deg"].as_f64().unwrap(),
    );
    let d = ((lat - 40.8765).to_radians() * 6_371_000.0)
        .hypot((lon + 73.525).to_radians() * 6_371_000.0 * 40.8765f64.to_radians().cos());
    assert!(d < 200.0, "{d} m");
    assert!(
        j["model"]["atmosphere"]
            .as_str()
            .is_some_and(|a| a.contains("still air")),
        "{}",
        j["model"]
    );
    assert!(j["model"]["wind_fit_enu"].is_array(), "{}", j["model"]);
    assert!(stdout.contains("wind: nuisance fit"), "{stdout}");
    let html = std::fs::read_to_string(out.join("locate.html")).unwrap();
    assert!(html.contains("nuisance"), "html names the wind fit");
    assert!(
        html.contains("weather: not used for simulator input"),
        "html names the weather source"
    );
    assert!(
        html.contains("<svg") && !html.contains("<script") && !html.contains("src=\"http"),
        "self-contained"
    );
}

#[test]
fn locate_no_events_exit_3() {
    let dir = tempfile::tempdir().unwrap();
    let (_, truth) = sim(dir.path());
    let (code, stdout, stderr) = locate(
        dir.path(),
        &fixture("plain.wav"),
        &[
            "--area",
            "40.8840,-73.5110~5km",
            "--when",
            "2026-09-08T20:40:00Z±5s",
            "--sim-truth",
            truth.to_str().unwrap(),
            "--out",
            dir.path().join("r").to_str().unwrap(),
        ],
    );
    assert_eq!(code, 3, "{stdout}\n{stderr}");
    assert!(
        stdout.contains("inconclusive") && stdout.contains("no Doppler events"),
        "{stdout}"
    );
}

#[test]
fn locate_no_match_shows_no_pin() {
    let dir = tempfile::tempdir().unwrap();
    let (wav, truth) = sim(dir.path());
    // Right place, wrong hour: nothing matches.
    let (code, stdout, stderr) = locate(
        dir.path(),
        &wav,
        &[
            "--area",
            "40.8840,-73.5110~5km",
            "--when",
            "2026-09-08T18:40:00Z±5s",
            "--sim-truth",
            truth.to_str().unwrap(),
            "--out",
            dir.path().join("r").to_str().unwrap(),
        ],
    );
    assert_eq!(code, 3, "{stdout}\n{stderr}");
    assert!(stdout.contains("inconclusive"), "{stdout}");
    assert!(
        !stdout.contains("best:") && !stdout.contains("openstreetmap"),
        "no pin on inconclusive: {stdout}"
    );
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("r/locate.json")).unwrap()).unwrap();
    assert_eq!(j["verdict_kind"], "inconclusive");
    let html = std::fs::read_to_string(dir.path().join("r/locate.html")).unwrap();
    assert!(!html.contains("openstreetmap"), "no pin in the HTML either");
}

#[test]
fn locate_missing_area_non_interactive() {
    // Not a terminal (assert_cmd pipes stdin): no prompt, a clear error instead.
    let dir = tempfile::tempdir().unwrap();
    let (code, stdout, stderr) = locate(
        dir.path(),
        &fixture("plain.wav"),
        &["--when", "2026-09-08T20:40:00Z±5s"],
    );
    assert_ne!(code, 0, "{stdout}");
    assert!(
        stderr.contains("--area") && stderr.contains("terminal"),
        "{stderr}"
    );
}

fn locate_with_data(data: &Path, file: &Path, args: &[&str]) -> (i32, String, String) {
    let o = common::geomedia()
        .env("GEOMEDIA_DATA", data)
        .arg("locate")
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

fn run_dirs(data: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(data.join("runs"))
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    v.sort();
    v
}

#[test]
fn locate_writes_managed_run() {
    let dir = tempfile::tempdir().unwrap();
    let (wav, truth) = sim(dir.path());
    let data = dir.path().join("data");
    let args = [
        "--area",
        "40.8840,-73.5110~5km",
        "--when",
        "2026-09-08T20:40:00Z±5s",
        "--sim-truth",
        truth.to_str().unwrap(),
    ];
    let (code, stdout, stderr) = locate_with_data(&data, &wav, &args);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let runs = run_dirs(&data);
    assert_eq!(runs.len(), 1, "{runs:?}");
    let run = &runs[0];
    let name = run.file_name().unwrap().to_string_lossy().into_owned();
    assert!(name.contains("_locate_audio"), "{name}");
    for f in ["locate.json", "locate.html", "events.json", "run.json"] {
        assert!(run.join(f).exists(), "{f} missing in {run:?}");
    }
    let info: serde_json::Value =
        serde_json::from_slice(&std::fs::read(run.join("run.json")).unwrap()).unwrap();
    assert_eq!(info["kind"], "locate");
    assert_eq!(info["verdict"], "located");
    assert!(
        info["file"].as_str().unwrap().ends_with("audio.wav"),
        "{info}"
    );
    assert!(
        stdout.contains(&run.display().to_string()),
        "run path printed: {stdout}"
    );
}

#[test]
fn out_also_copies() {
    let dir = tempfile::tempdir().unwrap();
    let (wav, truth) = sim(dir.path());
    let data = dir.path().join("data");
    let out = dir.path().join("mine");
    let args = [
        "--area",
        "40.8840,-73.5110~5km",
        "--when",
        "2026-09-08T20:40:00Z±5s",
        "--sim-truth",
        truth.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ];
    let (code, stdout, stderr) = locate_with_data(&data, &wav, &args);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    assert!(out.join("locate.json").exists() && out.join("locate.html").exists());
    let runs = run_dirs(&data);
    assert_eq!(runs.len(), 1, "{runs:?}");
    assert!(runs[0].join("locate.json").exists());
}

#[test]
fn locate_sim_quiet_found_blind() {
    // suite/quiet: the far PC12 pass is too faint for free discovery; guided scoring finds it
    // and the two passes locate the recording. Without guidance it can't be located.
    let dir = tempfile::tempdir().unwrap();
    let o = common::geomedia()
        .arg("sim")
        .arg(fixture("sim/suite/quiet.toml"))
        .arg("--out")
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(o.status.success());
    let (wav, truth) = (dir.path().join("audio.wav"), dir.path().join("truth.json"));
    let args = |out: &Path| -> Vec<String> {
        [
            "--area",
            "40.8900,-73.4900~6km",
            "--when",
            "2026-09-08T21:30:00Z±5s",
            "--sim-truth",
            truth.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    };
    let with = dir.path().join("with");
    let a = args(&with);
    let a: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, stdout, stderr) = locate(dir.path(), &wav, &a);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let ev: serde_json::Value =
        serde_json::from_slice(&std::fs::read(with.join("events.json")).unwrap()).unwrap();
    // The far, faint pass is now found from the audio alone (blind templates): an unguided
    // event near its arrival (56.8 s), so guidance adds no duplicate.
    assert!(
        ev["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["guided"] == false && (e["fit"]["tc"].as_f64().unwrap() - 56.8).abs() < 3.0),
        "{ev:#}"
    );
    let without = dir.path().join("without");
    let mut b = args(&without);
    b.push("--no-guided".into());
    let b: Vec<&str> = b.iter().map(String::as_str).collect();
    let (code, stdout, stderr) = locate(dir.path(), &wav, &b);
    assert_eq!(code, 0, "located without guidance too: {stdout}\n{stderr}");
}
