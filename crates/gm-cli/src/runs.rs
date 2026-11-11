//! Managed run folders: every report a command writes goes into its own folder under the kept
//! data's `runs/`, with a `run.json` saying what was run, so `geomedia data` can list it.

use std::path::{Path, PathBuf};

use anyhow::Result;

#[derive(Debug, Default, serde::Serialize)]
pub struct RunInfo {
    pub kind: String,
    pub file: String,
    pub area: String,
    pub when: String,
    pub verdict: String,
}

/// A new run folder under `base`: `<UTC time>_<kind>[_<file stem>]`, with `-2`, `-3`, … when
/// that name is taken (created atomically, so two runs never share a folder).
pub fn new_run_in(base: &Path, kind: &str, file: Option<&Path>) -> Result<PathBuf> {
    std::fs::create_dir_all(base)?;
    let t = time::OffsetDateTime::now_utc();
    let stamp = format!(
        "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}",
        t.year(),
        u8::from(t.month()),
        t.day(),
        t.hour(),
        t.minute(),
        t.second()
    );
    let stem = file
        .and_then(|f| f.file_stem())
        .map(|s| s.to_string_lossy().replace(['/', '\\', ':'], "_"));
    let name = match stem {
        Some(s) => format!("{stamp}_{kind}_{s}"),
        None => format!("{stamp}_{kind}"),
    };
    for n in 1..1000 {
        let dir = if n == 1 {
            base.join(&name)
        } else {
            base.join(format!("{name}-{n}"))
        };
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("no free run folder name for {name}")
}

/// A new run folder in the kept data's `runs/`.
pub fn new_run(kind: &str, file: Option<&Path>) -> Result<PathBuf> {
    new_run_in(&gm_sources::paths::runs_dir(), kind, file)
}

/// Write `run.json` (what was run, when, the verdict) into `dir`.
pub fn finish_run(dir: &Path, info: &RunInfo) -> Result<()> {
    let mut v = serde_json::to_value(info)?;
    v["created"] = serde_json::json!(
        time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_default()
    );
    v["tool_version"] = serde_json::json!(env!("CARGO_PKG_VERSION"));
    std::fs::write(dir.join("run.json"), serde_json::to_vec_pretty(&v)?)?;
    Ok(())
}

/// Write `name` into the managed run folder and, when given, the user's `--out` folder too.
pub fn write_both(run: &Path, out: Option<&Path>, name: &str, bytes: &[u8]) -> Result<()> {
    std::fs::write(run.join(name), bytes)?;
    if let Some(o) = out {
        std::fs::create_dir_all(o)?;
        std::fs::write(o.join(name), bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_dirs_never_collide() {
        let base = tempfile::tempdir().unwrap();
        let f = Path::new("/x/clip one.wav");
        let a = new_run_in(base.path(), "locate", Some(f)).unwrap();
        let b = new_run_in(base.path(), "locate", Some(f)).unwrap();
        assert_ne!(a, b);
        assert!(a.is_dir() && b.is_dir());
        let name = a.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.ends_with("_locate_clip one"), "{name}");
        assert!(new_run_in(base.path(), "eval", None).unwrap().is_dir());
    }
}
