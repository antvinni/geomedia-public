mod common;

#[test]
fn version_prints() -> Result<(), Box<dyn std::error::Error>> {
    let out = common::geomedia().arg("--version").output()?;
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout)?, "geomedia 0.1.0\n");
    Ok(())
}

use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn meta(args: &[&str], file: &Path, out: &Path) -> Run {
    let o = common::geomedia()
        .arg("meta")
        .arg(file)
        .arg("--out")
        .arg(out)
        .args(args)
        .output()
        .unwrap();
    Run {
        code: o.status.code().unwrap_or(-1),
        stdout: String::from_utf8(o.stdout).unwrap(),
        stderr: String::from_utf8(o.stderr).unwrap(),
    }
}

#[test]
fn meta_phone_gps() {
    let dir = tempfile::tempdir().unwrap();
    let r = meta(&[], &fixture("phone_gps.mov"), dir.path());
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(r.stdout.contains("Metadata (untrusted)"), "{}", r.stdout);
    assert!(r.stdout.contains("51.507200"), "{}", r.stdout);
    assert!(r.stdout.contains("iPhone 15"), "{}", r.stdout);
    assert!(dir.path().join("meta.html").is_file());
    assert!(dir.path().join("meta.json").is_file());
    // The managed run's report comes first; `--out` gets a copy.
    let last = r.stdout.lines().last().unwrap();
    assert_eq!(
        last,
        format!("copy:   {}", dir.path().join("meta.html").display())
    );
    let report = r
        .stdout
        .lines()
        .find_map(|l| l.strip_prefix("report: "))
        .expect("report line");
    assert!(report.contains("_meta_phone_gps"), "{report}");
    assert!(std::path::Path::new(report).is_file(), "{report}");
}

#[test]
fn meta_reencoded() {
    let dir = tempfile::tempdir().unwrap();
    let r = meta(&[], &fixture("reencoded.mp4"), dir.path());
    assert_eq!(r.code, 0);
    assert!(
        !r.stdout
            .lines()
            .any(|l| l.trim_start().starts_with("location")),
        "{}",
        r.stdout
    );
    assert!(
        r.stdout
            .lines()
            .any(|l| l.contains("platform_trace") && l.contains("ffmpeg")),
        "{}",
        r.stdout
    );
}

#[test]
fn meta_audio_only() {
    let dir = tempfile::tempdir().unwrap();
    let r = meta(&[], &fixture("plain.wav"), dir.path());
    assert_eq!(r.code, 0);
    assert!(
        r.stdout.lines().any(|l| l == "metadata: empty"),
        "{}",
        r.stdout
    );
    let html = std::fs::read_to_string(dir.path().join("meta.html")).unwrap();
    assert!(html.contains("Metadata: empty"));
}

#[test]
fn meta_unrecognised() {
    let dir = tempfile::tempdir().unwrap();
    let r = meta(&[], &fixture("hello.bin"), dir.path());
    assert_eq!(r.code, 0);
    assert!(r.stdout.contains("unrecognised"), "{}", r.stdout);
    assert!(
        r.stdout.lines().any(|l| l == "metadata: empty"),
        "{}",
        r.stdout
    );
}

#[test]
fn meta_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let r = meta(&[], Path::new("/definitely/not/here.mov"), dir.path());
    assert_eq!(r.code, 1);
    assert!(r.stderr.starts_with("error:"), "{}", r.stderr);
}

#[test]
fn meta_json_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let r = meta(
        &["--json", "-", "--no-report"],
        &fixture("phone_gps.mov"),
        dir.path(),
    );
    assert_eq!(r.code, 0, "{}", r.stderr);
    let v: serde_json::Value = serde_json::from_str(&r.stdout).unwrap();
    assert_eq!(v["schema_version"], 1);
    assert!(!dir.path().join("meta.html").exists());
}

#[test]
fn meta_hostile_file_name_is_escaped() {
    let dir = tempfile::tempdir().unwrap();
    let evil = dir.path().join("evil\x1b[2J\x1b]0;x\x07.wav");
    std::fs::copy(fixture("plain.wav"), &evil).unwrap();
    let r = meta(&[], &evil, &dir.path().join("out"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(
        !r.stdout.contains('\x1b') && !r.stdout.contains('\x07'),
        "{:?}",
        r.stdout
    );
    let missing = dir.path().join("missing\x1b[31m.mov");
    let r = meta(&[], &missing, &dir.path().join("out2"));
    assert_eq!(r.code, 1);
    assert!(!r.stderr.contains('\x1b'), "{:?}", r.stderr);
}
