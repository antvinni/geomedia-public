//! Data sources for geomedia: a visible registry, a local cache with resumable downloads, and
//! the readers for each source (ADS-B archives, aircraft types, terrain, geoid, geocoding).

pub mod adsblol;
pub mod area;
pub mod cache;
pub mod clean;
pub mod geocode;
pub mod inventory;
pub mod paths;
pub mod query;
pub mod registry;
pub mod terrain;
pub mod trace;
pub mod types;
pub mod weather;

pub use cache::{Fetch, cache_dir, fetch, user_agent};

/// Held by every test that sets GEOMEDIA_* variables, so none sees another's value (or none)
/// and falls through to the real folders.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
pub use registry::{SourceInfo, registry};

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("{path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("download {url}: {reason}")]
    Http { url: String, reason: String },
    #[error("download {url}: HTTP {status}")]
    HttpStatus { url: String, status: u16 },
    #[error("{0}")]
    Invalid(String),
    #[error("{what} needs a {size} download; run with --yes to allow it")]
    NeedsConsent { what: String, size: String },
}

/// "3.9 GB", "412 MB", "18 KB".
pub fn human_size(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.1} GB", n as f64 / f64::from(1u32 << 30)),
        n if n >= 1 << 20 => format!("{:.0} MB", n as f64 / f64::from(1u32 << 20)),
        n => format!("{:.0} KB", n as f64 / 1024.0),
    }
}
