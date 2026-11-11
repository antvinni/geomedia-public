//! Absolute time as integer nanoseconds since the UNIX epoch. Differences between epochs are
//! exact; only the final relative-seconds value is f64.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Epoch {
    pub ns: i64,
}

impl Epoch {
    /// From UNIX seconds; the f64 input limits precision to ~0.24 µs near 1.7e9.
    pub fn from_unix_f64(s: f64) -> Epoch {
        Epoch {
            ns: (s * 1e9).round() as i64,
        }
    }

    pub fn from_rfc3339(s: &str) -> Option<Epoch> {
        let t = OffsetDateTime::parse(s.trim(), &Rfc3339).ok()?;
        Some(Epoch {
            ns: i64::try_from(t.unix_timestamp_nanos()).ok()?,
        })
    }

    /// RFC 3339 in UTC with the shortest exact fraction.
    pub fn to_rfc3339(self) -> String {
        OffsetDateTime::from_unix_timestamp_nanos(i128::from(self.ns))
            .ok()
            .and_then(|t| t.format(&Rfc3339).ok())
            .unwrap_or_else(|| format!("{}ns", self.ns))
    }

    /// Seconds from `origin` to `self` (exact integer difference, then f64).
    pub fn seconds_since(self, origin: Epoch) -> f64 {
        (i128::from(self.ns) - i128::from(origin.ns)) as f64 * 1e-9
    }

    pub fn plus_seconds(self, s: f64) -> Epoch {
        Epoch {
            ns: self.ns.saturating_add((s * 1e9).round() as i64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_precision() {
        let e = Epoch::from_rfc3339("2026-09-08T20:17:29.662Z").unwrap();
        assert_eq!(e.to_rfc3339(), "2026-09-08T20:17:29.662Z");
        let f = Epoch { ns: e.ns + 1 };
        assert!((f.seconds_since(e) - 1e-9).abs() < 1e-12);
        assert!((e.plus_seconds(920.7).seconds_since(e) - 920.7).abs() < 1e-9);
        assert_eq!(Epoch::from_unix_f64(1.5).ns, 1_500_000_000);
        assert!(Epoch::from_rfc3339("not a time").is_none());
    }
}
