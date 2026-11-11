//! `geomedia data clean`: pick items from the inventory and remove them, only inside
//! geomedia's own folders, never following a symlink out of them.

use std::path::{Path, PathBuf};

use crate::SourceError;
use crate::inventory::{Inventory, tree_bytes};

/// What to clean: every kind, or one kind, optionally narrowed to one item by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selector {
    /// `all`, or one of [`KINDS`].
    pub what: String,
    /// One item's name within the kind: a date, a tile (N40W074), a run id, a media name.
    pub item: Option<String>,
}

/// The kinds `geomedia data` lists.
pub const KINDS: [&str; 9] = [
    "adsb", "terrain", "geoid", "weather", "types", "geocode", "runs", "media", "partial",
];

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CleanItem {
    /// `kind name`, as listed.
    pub label: String,
    pub kind: String,
    pub path: PathBuf,
    pub bytes: u64,
}

/// What a clean would remove, and what it left alone on purpose.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct CleanPlan {
    pub items: Vec<CleanItem>,
    /// Items the selection covered but left alone, each `label: reason`: an ADS-B day with a
    /// download in progress (removed only when named), interrupted downloads under `all`
    /// (removed with `partial`), recordings under `all --unused-for` (by age only with `media`).
    pub skipped: Vec<String>,
}

impl Selector {
    pub fn parse(what: &str, item: Option<&str>) -> Result<Selector, SourceError> {
        if what != "all" && !KINDS.contains(&what) {
            return Err(SourceError::Invalid(format!(
                "unknown kind {what:?}: use all or one of {}",
                KINDS.join(", ")
            )));
        }
        Ok(Selector {
            what: what.to_string(),
            item: item.map(str::to_string),
        })
    }
}

/// The items `sel` names, oldest use first, keeping those used within `unused_for_s` seconds
/// (an item with no usage stamp counts by its modification time; one with neither is kept).
/// A named item that doesn't exist is an error, so a typo never reads as "cleaned".
pub fn plan_clean(
    inv: &Inventory,
    sel: &Selector,
    unused_for_s: Option<i64>,
) -> Result<CleanPlan, SourceError> {
    let now = time::OffsetDateTime::now_utc().unix_timestamp();
    let named: Vec<_> = inv
        .items
        .iter()
        .filter(|i| sel.what == "all" || i.kind == sel.what)
        .filter(|i| sel.item.as_ref().is_none_or(|n| &i.name == n))
        .collect();
    if let Some(n) = &sel.item
        && named.is_empty()
    {
        return Err(SourceError::Invalid(format!(
            "no {} item named {n:?} (`geomedia data` lists them)",
            sel.what
        )));
    }
    // Another geomedia may be downloading into an incomplete day: only an explicit name (or
    // `partial`, for the .part files alone) removes it.
    // A recording can't be re-downloaded: age alone removes it only when `media` is asked for.
    let hold = |i: &crate::inventory::Item| -> Option<&str> {
        if sel.item.is_some() {
            return None;
        }
        let incomplete = || {
            inv.adsb_days
                .iter()
                .any(|d| d.date == i.name && d.state == crate::inventory::DayState::Incomplete)
        };
        match (i.kind.as_str(), sel.what.as_str()) {
            ("adsb", _) if incomplete() => {
                Some("download may be in progress (name it, or clean partial)")
            }
            ("partial", "all") => Some("interrupted download (clean partial removes these)"),
            ("media", "all") if unused_for_s.is_some() => {
                Some("recording (by age only with `clean media --unused-for`)")
            }
            _ => None,
        }
    };
    let mut skipped = Vec::new();
    let named: Vec<_> = named
        .into_iter()
        .filter(|i| match hold(i) {
            Some(why) => {
                skipped.push(format!("{} {}: {why}", i.kind, i.name));
                false
            }
            None => true,
        })
        .collect();
    let mut v: Vec<(Option<i64>, CleanItem)> = named
        .into_iter()
        .filter(|i| match unused_for_s {
            None => true,
            Some(s) => i.last_used.is_some_and(|t| now - t >= s),
        })
        .map(|i| {
            (
                i.last_used,
                CleanItem {
                    label: format!("{} {}", i.kind, i.name),
                    kind: i.kind.clone(),
                    path: i.path.clone(),
                    bytes: i.bytes,
                },
            )
        })
        .collect();
    // A partial download inside an ADS-B day goes with the day; listing it twice would
    // count its bytes twice.
    let dirs: Vec<PathBuf> = v
        .iter()
        .filter(|(_, c)| c.kind != "partial")
        .map(|(_, c)| c.path.clone())
        .collect();
    v.retain(|(_, c)| !dirs.iter().any(|d| c.path != *d && c.path.starts_with(d)));
    v.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.label.cmp(&b.1.label)));
    Ok(CleanPlan {
        items: v.into_iter().map(|(_, c)| c).collect(),
        skipped,
    })
}

/// `p` resolved without following its last component, if it lies strictly inside one of
/// `bases` (each resolved too).
fn inside(p: &Path, bases: &[PathBuf]) -> Option<PathBuf> {
    let name = p.file_name()?;
    if name == ".." || name == "." {
        return None;
    }
    let full = p.parent()?.canonicalize().ok()?.join(name);
    bases
        .iter()
        .filter_map(|b| b.canonicalize().ok())
        .any(|b| full.starts_with(&b) && full != b)
        .then_some(full)
}

/// Remove each item, after checking it lies inside one of `bases`. A symlink is removed
/// itself, never its target; directories are removed without following links inside them.
/// Returns the bytes freed. Stops at the first item outside the bases or that fails.
pub fn execute_clean(items: &[CleanItem], bases: &[PathBuf]) -> Result<u64, SourceError> {
    let mut freed = 0;
    for it in items {
        let io = |source| SourceError::Io {
            path: it.path.display().to_string(),
            source,
        };
        let Some(p) = inside(&it.path, bases) else {
            // Already gone (an earlier item contained it) is fine; anything else is refused.
            if std::fs::symlink_metadata(&it.path).is_err() {
                continue;
            }
            return Err(SourceError::Invalid(format!(
                "refusing to delete {}: not inside geomedia's folders",
                it.path.display()
            )));
        };
        let Ok(m) = std::fs::symlink_metadata(&p) else {
            continue;
        };
        let bytes = tree_bytes(&p);
        if m.is_dir() {
            std::fs::remove_dir_all(&p).map_err(io)?;
        } else {
            std::fs::remove_file(&p).map_err(io)?;
        }
        freed += bytes;
    }
    Ok(freed)
}

/// `p` with symlinks resolved in its longest existing ancestor (the rest appended as is).
fn resolved(p: &Path) -> PathBuf {
    let abs = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let mut rest = Vec::new();
    let mut cur = abs.as_path();
    loop {
        if let Ok(c) = cur.canonicalize() {
            return rest.iter().rev().fold(c, |acc, n| acc.join(n));
        }
        match (cur.parent(), cur.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                cur = parent;
            }
            _ => return abs,
        }
    }
}

/// Refuse to clean when the folders could hold things that aren't geomedia's: a base at the
/// filesystem root or the home folder, the two bases the same, or one inside the other.
pub fn check_bases(cache: &Path, data: &Path, home: Option<&Path>) -> Result<(), SourceError> {
    let (c, d) = (resolved(cache), resolved(data));
    let home = home.map(resolved);
    let refuse = |why: &str| {
        Err(SourceError::Invalid(format!(
            "refusing to clean: {why} (source data {}, kept data {}); set GEOMEDIA_CACHE and \
             GEOMEDIA_DATA to folders of geomedia's own",
            c.display(),
            d.display()
        )))
    };
    for b in [&c, &d] {
        if b.parent().is_none() {
            return refuse("a folder is the filesystem root");
        }
        if home.as_ref() == Some(b) {
            return refuse("a folder is the home folder");
        }
    }
    if c.starts_with(&d) || d.starts_with(&c) {
        return refuse("the two folders overlap");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(p: &Path) -> CleanItem {
        CleanItem {
            label: "x".into(),
            kind: "geoid".into(),
            path: p.to_path_buf(),
            bytes: 0,
        }
    }

    #[test]
    fn clean_refuses_outside_bases() {
        let base = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let f = other.path().join("f");
        std::fs::write(&f, b"x").unwrap();
        let bases = vec![base.path().to_path_buf()];
        assert!(execute_clean(&[item(&f)], &bases).is_err());
        assert!(f.is_file());
        // The base itself and `..` escapes are refused too.
        assert!(execute_clean(&[item(base.path())], &bases).is_err());
        assert!(base.path().is_dir());
        let sneaky = base
            .path()
            .join("..")
            .join(other.path().file_name().unwrap());
        assert!(execute_clean(&[item(&sneaky)], &bases).is_err());
        assert!(f.is_file());
    }

    #[test]
    fn clean_removes_link_not_target() {
        let base = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let f = other.path().join("f");
        std::fs::write(&f, b"keep").unwrap();
        let link = base.path().join("l");
        std::os::unix::fs::symlink(&f, &link).unwrap();
        // A directory inside the base holding a link out: the link goes, the target stays.
        let d = base.path().join("d");
        std::fs::create_dir(&d).unwrap();
        std::os::unix::fs::symlink(other.path(), d.join("out")).unwrap();
        execute_clean(&[item(&link), item(&d)], &[base.path().to_path_buf()]).unwrap();
        assert!(std::fs::symlink_metadata(&link).is_err());
        assert!(!d.exists());
        assert_eq!(std::fs::read(&f).unwrap(), b"keep");
    }

    #[test]
    fn unsafe_bases_refused() {
        let b = tempfile::tempdir().unwrap();
        let (c, d) = (b.path().join("cache"), b.path().join("data"));
        assert!(check_bases(&c, &d, Some(Path::new("/home/x"))).is_ok());
        assert!(check_bases(Path::new("/"), &d, None).is_err());
        assert!(check_bases(&c, &c, None).is_err(), "same folder");
        assert!(
            check_bases(b.path(), &d, None).is_err(),
            "data inside cache"
        );
        assert!(check_bases(&c, &c.join("kept"), None).is_err());
        assert!(
            check_bases(&c, b.path(), None).is_err(),
            "cache inside data"
        );
        assert!(check_bases(&c, &d, Some(&c)).is_err(), "home folder");
    }

    #[test]
    fn unused_for_all_spares_recordings() {
        let old = time::OffsetDateTime::now_utc().unix_timestamp() - 200 * 86_400;
        let it = |kind: &str| crate::inventory::Item {
            kind: kind.into(),
            name: "x".into(),
            path: PathBuf::from(format!("/b/{kind}/x")),
            bytes: 1,
            last_used: Some(old),
            detail: String::new(),
        };
        let inv = Inventory {
            items: vec![it("media"), it("geoid")],
            ..Default::default()
        };
        let all = Selector::parse("all", None).unwrap();
        let p = plan_clean(&inv, &all, Some(90 * 86_400)).unwrap();
        assert_eq!(p.items.len(), 1, "{p:?}");
        assert_eq!(p.items[0].kind, "geoid");
        assert!(p.skipped.iter().any(|s| s.starts_with("media")), "{p:?}");
        let media = Selector::parse("media", None).unwrap();
        assert_eq!(
            plan_clean(&inv, &media, Some(90 * 86_400))
                .unwrap()
                .items
                .len(),
            1
        );
    }

    #[test]
    fn clean_unknown_kind_and_item() {
        assert!(Selector::parse("everything", None).is_err());
        let sel = Selector::parse("runs", Some("nope")).unwrap();
        assert!(plan_clean(&Inventory::default(), &sel, None).is_err());
        let all = Selector::parse("all", None).unwrap();
        assert_eq!(
            plan_clean(&Inventory::default(), &all, None).unwrap(),
            CleanPlan::default()
        );
    }
}
