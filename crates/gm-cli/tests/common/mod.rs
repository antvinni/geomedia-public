//! Shared by the CLI tests: every run of the binary is isolated from the user's real data.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

/// One private temporary base per test binary, kept for the process's lifetime.
fn base() -> &'static PathBuf {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().to_path_buf();
        std::mem::forget(d);
        p
    })
}

/// `geomedia`, with GEOMEDIA_CACHE and GEOMEDIA_DATA in a private temporary folder. A test may
/// still set either again (a later `.env` wins).
pub fn geomedia() -> assert_cmd::Command {
    let mut c = assert_cmd::Command::cargo_bin("geomedia").unwrap();
    c.env("GEOMEDIA_CACHE", base().join("cache"))
        .env("GEOMEDIA_DATA", base().join("data"));
    c
}
