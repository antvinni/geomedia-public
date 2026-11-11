//! Where geomedia keeps things. Two base folders:
//! - **source data** (`cache_dir`): re-downloadable or rebuildable — ADS-B days and indexes,
//!   terrain, geoid, weather, aircraft types, geocoding replies;
//! - **kept data** (`data_dir`): not re-creatable — `runs/` (reports, events) and `media/`.
//!
//! Each carries `layout.json` (the layout version) and `usage.json` (when each item was last
//! used, for `geomedia data`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::SourceError;

pub const LAYOUT_VERSION: u32 = 1;

/// `$GEOMEDIA_DATA`, else the platform's application-data folder + `geomedia`.
pub fn data_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("GEOMEDIA_DATA").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        return home.join("Library/Application Support/geomedia");
    }
    std::env::var_os("XDG_DATA_HOME")
        .filter(|p| !p.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".local/share"))
        .join("geomedia")
}

pub fn runs_dir() -> PathBuf {
    data_dir().join("runs")
}

pub fn media_dir() -> PathBuf {
    data_dir().join("media")
}

const LAYOUT_FILE: &str = "layout.json";
const USAGE_FILE: &str = "usage.json";

fn io(p: &Path) -> impl Fn(std::io::Error) -> SourceError + '_ {
    move |source| SourceError::Io {
        path: p.display().to_string(),
        source,
    }
}

fn layout_version(base: &Path) -> u32 {
    std::fs::read(base.join(LAYOUT_FILE))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v["version"].as_u64())
        .map_or(0, |v| v as u32)
}

/// Move `src` into `dst`: a missing destination is moved whole; directories are merged entry by
/// entry; an existing destination file is kept and the source left in place (noted).
fn merge_move(src: &Path, dst: &Path, steps: &mut Vec<String>) {
    if !dst.exists() {
        if let Some(parent) = dst.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // Another volume (GEOMEDIA_DATA elsewhere): copy, check, then remove the source.
        let moved = match std::fs::rename(src, dst) {
            Err(e) if e.kind() == std::io::ErrorKind::CrossesDevices => copy_move(src, dst),
            r => r,
        };
        match moved {
            Ok(()) => steps.push(format!("moved {} → {}", src.display(), dst.display())),
            Err(e) => steps.push(format!(
                "warning: could not move {} ({e}); left in place",
                src.display()
            )),
        }
        return;
    }
    if src.is_dir() && dst.is_dir() {
        if let Ok(rd) = std::fs::read_dir(src) {
            for e in rd.flatten() {
                merge_move(&e.path(), &dst.join(e.file_name()), steps);
            }
        }
        let _ = std::fs::remove_dir(src); // only succeeds once empty
        return;
    }
    steps.push(format!(
        "warning: kept {} (already there); left {} in place",
        dst.display(),
        src.display()
    ));
}

/// Bring both folders to the current layout; one line per step taken or problem met (empty when
/// nothing to do). Problems start with `warning:`; every line is worth showing.
/// v0 → v1: recordings move from `<cache>/media` to `<data>/media`.
pub fn ensure_layout_in(cache: &Path, data: &Path) -> Vec<String> {
    let mut steps = Vec::new();
    if let Err(e) = layout_steps(cache, data, &mut steps) {
        steps.push(format!("warning: {e}"));
    }
    steps
}

fn layout_steps(cache: &Path, data: &Path, steps: &mut Vec<String>) -> Result<(), SourceError> {
    std::fs::create_dir_all(cache).map_err(io(cache))?;
    std::fs::create_dir_all(data).map_err(io(data))?;
    let old = cache.join("media");
    if layout_version(cache) < 1 && old.is_dir() {
        merge_move(&old, &data.join("media"), steps);
    }
    // Recordings left in the cache (a failed move, a conflict): the cache stays unmigrated, so
    // the move is retried — and this said — on every run until it's resolved.
    let stranded = std::fs::symlink_metadata(&old).is_ok();
    if stranded {
        steps.push(format!(
            "warning: recordings remain in {} (source data); move them into {} by hand if this \
             repeats",
            old.display(),
            data.join("media").display()
        ));
    }
    for base in [cache, data] {
        if stranded && base == cache {
            continue;
        }
        if layout_version(base) < LAYOUT_VERSION {
            let p = base.join(LAYOUT_FILE);
            std::fs::write(&p, format!("{{\"version\":{LAYOUT_VERSION}}}\n")).map_err(io(&p))?;
        }
    }
    Ok(())
}

/// `ensure_layout_in` on the real folders.
pub fn ensure_layout() -> Vec<String> {
    ensure_layout_in(&crate::cache_dir(), &data_dir())
}

/// If `file` lies in the kept media folder, stamp its recording (the folder or file directly
/// under `media/`) as used now.
pub fn touch_media(file: &Path) {
    let (Ok(f), Ok(m)) = (file.canonicalize(), media_dir().canonicalize()) else {
        return;
    };
    if let Some(first) = f.strip_prefix(&m).ok().and_then(|r| r.iter().next()) {
        touch_used("media", &first.to_string_lossy());
    }
}

/// Record that `item` of `kind` was used now, in `base`'s `usage.json`. Best effort: usage
/// stamps only inform `geomedia data`, so a failure is ignored.
pub fn touch_used_in(base: &Path, kind: &str, item: &str) {
    let p = base.join(USAGE_FILE);
    let mut v: serde_json::Value = std::fs::read(&p)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    let now = time::OffsetDateTime::now_utc().unix_timestamp();
    if !v[kind].is_object() {
        v[kind] = serde_json::json!({});
    }
    v[kind][item] = serde_json::json!(now);
    let tmp = base.join(format!("{USAGE_FILE}.tmp{}", std::process::id()));
    if std::fs::create_dir_all(base).is_ok()
        && std::fs::write(&tmp, v.to_string()).is_ok()
        && std::fs::rename(&tmp, &p).is_err()
    {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// `touch_used_in` the folder that holds `kind` (runs and media: kept data; else source data).
pub fn touch_used(kind: &str, item: &str) {
    let base = if matches!(kind, "runs" | "media") {
        data_dir()
    } else {
        crate::cache_dir()
    };
    touch_used_in(&base, kind, item);
}

/// (kind, item) → last-used UNIX seconds, from `base`'s `usage.json`.
pub fn last_used(base: &Path) -> HashMap<(String, String), i64> {
    let Some(v) = std::fs::read(base.join(USAGE_FILE))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    if let Some(kinds) = v.as_object() {
        for (kind, items) in kinds {
            for (item, t) in items.as_object().into_iter().flatten() {
                if let Some(t) = t.as_i64() {
                    out.insert((kind.clone(), item.clone()), t);
                }
            }
        }
    }
    out
}

/// Copy `src` to `dst` (links as links, never followed), check each file's size, then remove
/// the source. On an error the source keeps whatever wasn't copied yet.
fn copy_move(src: &Path, dst: &Path) -> std::io::Result<()> {
    let m = std::fs::symlink_metadata(src)?;
    if m.file_type().is_symlink() {
        std::os::unix::fs::symlink(std::fs::read_link(src)?, dst)?;
        return std::fs::remove_file(src);
    }
    if m.is_dir() {
        std::fs::create_dir_all(dst)?;
        for e in std::fs::read_dir(src)? {
            let e = e?;
            copy_move(&e.path(), &dst.join(e.file_name()))?;
        }
        return std::fs::remove_dir(src);
    }
    let n = std::fs::copy(src, dst)?;
    if n != m.len() || std::fs::metadata(dst)?.len() != m.len() {
        return Err(std::io::Error::other(format!(
            "copied {n} of {} bytes",
            m.len()
        )));
    }
    std::fs::remove_file(src)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(p: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn data_dir_honours_override() {
        // SAFETY: every test that sets GEOMEDIA_* holds ENV_LOCK.
        let _g = crate::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("GEOMEDIA_DATA");
        unsafe { std::env::set_var("GEOMEDIA_DATA", "/tmp/gm-test-data") };
        assert_eq!(data_dir(), PathBuf::from("/tmp/gm-test-data"));
        assert_eq!(runs_dir(), PathBuf::from("/tmp/gm-test-data/runs"));
        assert_eq!(media_dir(), PathBuf::from("/tmp/gm-test-data/media"));
        match prev {
            Some(v) => unsafe { std::env::set_var("GEOMEDIA_DATA", v) },
            None => unsafe { std::env::remove_var("GEOMEDIA_DATA") },
        }
    }

    #[test]
    fn migrates_media_once() {
        let (c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write(&c.path().join("media/oysterbay/raw.webm"), b"audio");
        write(&c.path().join("adsb/2026-09-08/index.json"), b"{}");
        let steps = ensure_layout_in(c.path(), d.path());
        assert!(!steps.is_empty(), "{steps:?}");
        assert_eq!(
            std::fs::read(d.path().join("media/oysterbay/raw.webm")).unwrap(),
            b"audio"
        );
        assert!(!c.path().join("media").exists());
        assert!(
            c.path().join("adsb/2026-09-08/index.json").exists(),
            "source data stays"
        );
        for base in [c.path(), d.path()] {
            let v: serde_json::Value =
                serde_json::from_slice(&std::fs::read(base.join("layout.json")).unwrap()).unwrap();
            assert_eq!(v["version"], LAYOUT_VERSION);
        }
        assert!(
            ensure_layout_in(c.path(), d.path()).is_empty(),
            "second call is a no-op"
        );
    }

    #[test]
    fn migration_is_idempotent() {
        // The destination already has this recording (an earlier, partial migration): keep it,
        // leave the cache copy for the user, don't fail.
        let (c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write(&c.path().join("media/oysterbay/raw.webm"), b"old");
        write(&d.path().join("media/oysterbay/raw.webm"), b"new");
        write(&c.path().join("media/other/a.wav"), b"a");
        let steps = ensure_layout_in(c.path(), d.path());
        assert_eq!(
            std::fs::read(d.path().join("media/oysterbay/raw.webm")).unwrap(),
            b"new"
        );
        assert!(c.path().join("media/oysterbay/raw.webm").exists());
        assert_eq!(
            std::fs::read(d.path().join("media/other/a.wav")).unwrap(),
            b"a"
        );
        assert!(steps.iter().any(|s| s.contains("kept")), "{steps:?}");
    }

    #[test]
    fn failed_migration_retries_and_reports() {
        use std::os::unix::fs::PermissionsExt;
        let (c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write(&c.path().join("media/oysterbay/raw.webm"), b"audio");
        let mode = |m| std::fs::Permissions::from_mode(m);
        std::fs::set_permissions(d.path(), mode(0o555)).unwrap();
        let steps = ensure_layout_in(c.path(), d.path());
        std::fs::set_permissions(d.path(), mode(0o755)).unwrap();
        assert!(
            steps.iter().any(|s| s.contains("could not move")),
            "the failed move is reported: {steps:?}"
        );
        assert_eq!(layout_version(c.path()), 0, "not marked migrated");
        // Next run, with the data folder writable: the move happens.
        let steps = ensure_layout_in(c.path(), d.path());
        assert_eq!(
            std::fs::read(d.path().join("media/oysterbay/raw.webm")).unwrap(),
            b"audio",
            "{steps:?}"
        );
        assert_eq!(layout_version(c.path()), LAYOUT_VERSION);
    }

    #[test]
    fn leftover_media_keeps_cache_unmigrated() {
        // A conflict leaves a recording in the cache: the cache stays at v0, so every run says so.
        let (c, d) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        write(&c.path().join("media/oysterbay/raw.webm"), b"old");
        write(&d.path().join("media/oysterbay/raw.webm"), b"new");
        ensure_layout_in(c.path(), d.path());
        assert_eq!(layout_version(c.path()), 0);
        assert!(
            !ensure_layout_in(c.path(), d.path()).is_empty(),
            "still reported"
        );
    }

    #[test]
    fn copy_move_keeps_links_and_removes_source() {
        let (s, t) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let src = s.path().join("rec");
        write(&src.join("a/raw.webm"), b"audio");
        std::os::unix::fs::symlink("/nonexistent/target", src.join("link")).unwrap();
        let dst = t.path().join("rec");
        copy_move(&src, &dst).unwrap();
        assert_eq!(std::fs::read(dst.join("a/raw.webm")).unwrap(), b"audio");
        assert_eq!(
            std::fs::read_link(dst.join("link")).unwrap(),
            Path::new("/nonexistent/target")
        );
        assert!(!src.exists());
    }

    #[test]
    fn usage_stamps_round_trip() {
        let b = tempfile::tempdir().unwrap();
        touch_used_in(b.path(), "adsb", "2026-09-08");
        touch_used_in(b.path(), "terrain", "N40W074");
        let u = last_used(b.path());
        let now = time::OffsetDateTime::now_utc().unix_timestamp();
        let t = u[&("adsb".to_string(), "2026-09-08".to_string())];
        assert!((now - t).abs() < 5, "{t} vs {now}");
        assert!(u.contains_key(&("terrain".to_string(), "N40W074".to_string())));
        assert!(last_used(Path::new("/nonexistent/x")).is_empty());
    }
}
