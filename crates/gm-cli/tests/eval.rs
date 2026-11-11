mod common;

use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .canonicalize()
        .unwrap()
}

fn eval(manifest: &Path, out: &Path) -> (i32, String, String) {
    let o = common::geomedia()
        .env("GEOMEDIA_CACHE", out)
        .arg("eval")
        .arg(manifest)
        .arg("--out")
        .arg(out)
        .output()
        .unwrap();
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8(o.stdout).unwrap(),
        String::from_utf8(o.stderr).unwrap(),
    )
}

#[test]
fn eval_sim_cases() {
    let dir = tempfile::tempdir().unwrap();
    let f = fixtures();
    let manifest = dir.path().join("m.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"
[[case]]
id = "two"
kind = "sim"
scenario = "{f}/sim/two_props.toml"
area = "40.8840,-73.5110~5km"
when = "2026-09-08T20:40:00Z±5s"
truth_note = "simulator"

[[case]]
id = "one"
kind = "sim"
scenario = "{f}/sim/one_pass.toml"
area = "40.8800,-73.5200~4km"
when = "2026-09-08T20:50:00Z±5s"
truth_note = "simulator"

[[case]]
id = "missing"
kind = "media"
media = "{f}/does-not-exist.webm"
area = "40.8800,-73.5200~4km"
when = "2026-09-08T20:50:00Z±5s"
truth = [40.8765, -73.525]
truth_note = "none"
"#,
            f = f.display()
        ),
    )
    .unwrap();
    let out = dir.path().join("out");
    let (code, stdout, stderr) = eval(&manifest, &out);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("results.json")).unwrap()).unwrap();
    let rows = rows["cases"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["verdict_kind"], "located");
    assert!(rows[0]["error_m"].as_f64().unwrap() < 200.0, "{}", rows[0]);
    // The passes the tracks say the recording holds (≤ 8 km), with their audio times: the
    // street-sound labels for a recording with ADS-B are made from these.
    let expected = rows[0]["expected_passes"].as_array().unwrap();
    assert_eq!(expected.len(), 2, "{}", rows[0]);
    assert!(
        expected
            .iter()
            .all(|p| p["t_audio"].as_f64().is_some_and(f64::is_finite)),
        "{}",
        rows[0]
    );
    assert_ne!(rows[1]["verdict_kind"], "located");
    assert_eq!(rows[2]["verdict_kind"], "error");
    let summary = std::fs::read_to_string(out.join("summary.md")).unwrap();
    let first = summary
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .unwrap();
    assert!(first.contains("confident wrong: 0"), "{summary}");
    assert!(
        summary.contains("located: 1 of 3") || summary.contains("located: 1/3"),
        "{summary}"
    );
}

#[test]
fn eval_counts_confident_wrong_after_error_row() {
    // Missing media first: later cases must still run. Then a sim case whose stated truth is
    // ~8 km from where the recording was made: a located answer there is confidently wrong.
    let dir = tempfile::tempdir().unwrap();
    let f = fixtures();
    let manifest = dir.path().join("m.toml");
    std::fs::write(
        &manifest,
        format!(
            r#"
[[case]]
id = "missing"
kind = "media"
media = "{f}/does-not-exist.webm"
area = "40.8800,-73.5200~4km"
when = "2026-09-08T20:50:00Z±5s"
truth = [40.8765, -73.525]
truth_note = "none"

[[case]]
id = "liar"
kind = "sim"
scenario = "{f}/sim/two_props.toml"
area = "40.8840,-73.5110~5km"
when = "2026-09-08T20:40:00Z±5s"
truth = [40.95, -73.525]
truth_note = "deliberately wrong truth"
"#,
            f = f.display()
        ),
    )
    .unwrap();
    let out = dir.path().join("out");
    let (code, stdout, stderr) = eval(&manifest, &out);
    assert_eq!(code, 0, "{stdout}\n{stderr}");
    let rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("results.json")).unwrap()).unwrap();
    assert_eq!(rows["cases"][0]["verdict_kind"], "error", "{rows}");
    assert_eq!(rows["cases"][1]["verdict_kind"], "located", "{rows}");
    let summary = std::fs::read_to_string(out.join("summary.md")).unwrap();
    let first = summary
        .lines()
        .find(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .unwrap();
    assert!(first.contains("confident wrong: 1"), "{summary}");
    assert!(first.contains("liar"), "{summary}");
}

#[test]
fn eval_rejects_unsafe_or_duplicate_ids() {
    let dir = tempfile::tempdir().unwrap();
    let f = fixtures();
    let case = |id: &str| {
        format!(
            "[[case]]\nid = \"{id}\"\nkind = \"sim\"\nscenario = \"{}/sim/one_pass.toml\"\narea = \"40.8800,-73.5200~4km\"\nwhen = \"2026-09-08T20:50:00Z±5s\"\ntruth_note = \"simulator\"\n",
            f.display()
        )
    };
    for (name, body) in [
        ("escape", case("../../escaped")),
        ("absolute", case("/tmp/geomedia-escape")),
        ("empty", case("")),
        ("dup", format!("{}{}", case("same"), case("same"))),
    ] {
        let manifest = dir.path().join(format!("{name}.toml"));
        std::fs::write(&manifest, body).unwrap();
        let out = dir.path().join(name).join("a").join("b");
        let (code, _, stderr) = eval(&manifest, &out);
        assert_eq!(code, 1, "{name}: {stderr}");
        assert!(stderr.contains("id"), "{name}: {stderr}");
        assert!(!dir.path().join(name).join("escaped").exists(), "{name}");
        assert!(!out.join("results.json").exists(), "{name}");
    }
}

#[test]
fn eval_relative_paths() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("eval")).unwrap();
    std::fs::copy(
        fixtures().join("sim/two_props.toml"),
        dir.path().join("two.toml"),
    )
    .unwrap();
    let manifest = dir.path().join("eval/m.toml");
    std::fs::write(&manifest, "[[case]]\nid = \"rel\"\nkind = \"sim\"\nscenario = \"../two.toml\"\narea = \"40.8840,-73.5110~5km\"\nwhen = \"2026-09-08T20:40:00Z±5s\"\ntruth_note = \"simulator\"\n").unwrap();
    let out = dir.path().join("out");
    let (code, _, stderr) = eval(&manifest, &out);
    assert_eq!(code, 0, "{stderr}");
    let rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("results.json")).unwrap()).unwrap();
    assert_eq!(rows["cases"][0]["verdict_kind"], "located", "{rows}");
}

/// The simulator suite: zero confident wrong answers; the multi-pass cases located within
/// 300 m (`windy` since the v2b wind fit, `quiet` since v2c-2's guided scoring finds its far
/// pass); the single pass never located. Slow (renders and searches six scenarios): `cargo test -- --ignored`.
#[test]
#[ignore = "slow: runs the whole simulator suite"]
fn sim_suite_has_no_confident_wrong() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = tempfile::tempdir().unwrap();
    common::geomedia()
        .env("GEOMEDIA_CACHE", out.path())
        .arg("eval")
        .arg(root.join("tests/eval/sim-suite.toml"))
        .arg("--out")
        .arg(out.path())
        .assert()
        .success();
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.path().join("results.json")).unwrap()).unwrap();
    let cases = v["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 7);
    let located_err = |c: &serde_json::Value| {
        (c["verdict_kind"] == "located").then(|| c["error_m"].as_f64().unwrap())
    };
    let wrong = cases
        .iter()
        .filter_map(located_err)
        .filter(|e| *e > 3000.0)
        .count();
    assert_eq!(wrong, 0, "{v:#}");
    for id in [
        "sim-crossing",
        "sim-three",
        "sim-decoy",
        "sim-decel",
        "sim-windy",
        "sim-quiet",
    ] {
        let c = cases.iter().find(|c| c["id"] == id).unwrap();
        let e = located_err(c);
        assert!(
            e.is_some_and(|e| e <= 300.0),
            "{id} not located within 300 m: {c:#}"
        );
    }
    let single = cases.iter().find(|c| c["id"] == "sim-single").unwrap();
    assert_ne!(single["verdict_kind"], "located", "{single:#}");
}

#[test]
fn media_prefix_resolves() {
    // `media:` names a recording in the kept-data media folder, whatever the machine.
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    let manifest = dir.path().join("m.toml");
    std::fs::write(
        &manifest,
        "[[case]]\nid = \"m\"\nkind = \"media\"\nmedia = \"media:x/y.wav\"\narea = \"40.88,-73.52~4km\"\nwhen = \"2026-09-08T20:50:00Z±5s\"\ntruth = [40.8765, -73.525]\ntruth_note = \"none\"\n",
    )
    .unwrap();
    let out = dir.path().join("out");
    let o = common::geomedia()
        .env("GEOMEDIA_DATA", &data)
        .arg("eval")
        .arg(&manifest)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    assert!(o.status.success());
    let rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("results.json")).unwrap()).unwrap();
    let msg = rows["cases"][0]["verdict"].as_str().unwrap();
    let want = data.join("media/x/y.wav");
    assert!(msg.contains(&want.display().to_string()), "{msg}");
}

#[test]
fn using_a_recording_stamps_it() {
    // A kept recording that eval opens counts as used, so `clean --unused-for` judges it by
    // use, not by file age.
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    std::fs::create_dir_all(data.join("media/x")).unwrap();
    std::fs::write(data.join("media/x/y.wav"), b"not audio").unwrap();
    let manifest = dir.path().join("m.toml");
    std::fs::write(
        &manifest,
        "[[case]]\nid = \"m\"\nkind = \"media\"\nmedia = \"media:x/y.wav\"\narea = \"40.88,-73.52~4km\"\nwhen = \"2026-09-08T20:50:00Z±5s\"\ntruth = [40.8765, -73.525]\ntruth_note = \"none\"\n",
    )
    .unwrap();
    let o = common::geomedia()
        .env("GEOMEDIA_DATA", &data)
        .arg("eval")
        .arg(&manifest)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let u: serde_json::Value =
        serde_json::from_slice(&std::fs::read(data.join("usage.json")).expect("usage.json"))
            .unwrap();
    assert!(u["media"]["x"].is_i64(), "{u}");
}
