//! Lenient ISO 8601 parsing for embedded timestamps, plus a plausibility check.

use time::format_description::well_known::Rfc3339;
use time::{Date, OffsetDateTime, PrimitiveDateTime, UtcOffset};

use crate::model::{Hint, HintKind, HintValue};

/// Earliest timestamp considered plausible for a digital recording.
pub const PLAUSIBLE_FROM_UNIX: i64 = 631_152_000; // 1990-01-01T00:00:00Z

/// A parsed timestamp: its RFC 3339 rendering (offset kept as recorded; none if local)
/// and the instant used for the plausibility check (local times treated as UTC).
pub struct Parsed {
    pub rfc3339: String,
    pub instant: OffsetDateTime,
}

/// Parse "2026-05-14T10:02:00-0700", "...-07:00", "...Z", "2026-05-14 10:02:00" (local) or "2026-05-14".
pub fn parse_iso8601(s: &str) -> Option<Parsed> {
    let s = s.trim();
    let date = parse_date(s.get(..10)?)?;
    let rest = &s[10..];
    if rest.is_empty() {
        return Some(Parsed {
            rfc3339: s[..10].to_string(),
            instant: date.midnight().assume_utc(),
        });
    }
    let rest = rest.strip_prefix(['T', ' '])?;
    let hms = rest.get(..8)?;
    let (hour, minute, second) = parse_hms(hms)?;
    let mut tail = &rest[8..];
    let mut nanos = 0u32;
    let mut frac = "";
    if let Some(f) = tail.strip_prefix(['.', ',']) {
        let digits = f.find(|c: char| !c.is_ascii_digit()).unwrap_or(f.len());
        if digits == 0 || digits > 9 {
            return None;
        }
        frac = &f[..digits];
        nanos = format!("{frac:0<9}").parse().ok()?;
        tail = &f[digits..];
    }
    let local = PrimitiveDateTime::new(
        date,
        time::Time::from_hms_nano(hour, minute, second, nanos).ok()?,
    );
    let offset = match tail {
        "" => None,
        "Z" | "z" => Some(UtcOffset::UTC),
        t => Some(parse_offset(t)?),
    };
    Some(match offset {
        None => {
            let frac = if frac.is_empty() {
                String::new()
            } else {
                format!(".{frac}")
            };
            Parsed {
                rfc3339: format!("{}T{hms}{frac}", &s[..10]),
                instant: local.assume_utc(),
            }
        }
        Some(off) => {
            let t = local.assume_offset(off);
            Parsed {
                rfc3339: t.format(&Rfc3339).ok()?,
                instant: t,
            }
        }
    })
}

fn parse_date(s: &str) -> Option<Date> {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let year: i32 = digits(&s[..4])?;
    let month: u8 = digits(&s[5..7])?;
    let day: u8 = digits(&s[8..10])?;
    Date::from_calendar_date(year, time::Month::try_from(month).ok()?, day).ok()
}

fn parse_hms(s: &str) -> Option<(u8, u8, u8)> {
    let b = s.as_bytes();
    if b.len() != 8 || b[2] != b':' || b[5] != b':' {
        return None;
    }
    Some((digits(&s[..2])?, digits(&s[3..5])?, digits(&s[6..8])?))
}

/// "+07:00", "-0700" or "+07".
fn parse_offset(s: &str) -> Option<UtcOffset> {
    let sign: i8 = match s.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let body = s[1..].replace(':', "");
    if !body.is_ascii() {
        return None;
    }
    let (h, m): (i8, i8) = match body.len() {
        2 => (digits(&body)?, 0),
        4 => (digits(&body[..2])?, digits(&body[2..])?),
        _ => return None,
    };
    UtcOffset::from_hms(sign * h, sign * m, 0).ok()
}

fn digits<T: std::str::FromStr>(s: &str) -> Option<T> {
    if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Whether `t` lies between 1990-01-01 and one day after now.
pub fn plausible(t: OffsetDateTime) -> bool {
    let from = OffsetDateTime::from_unix_timestamp(PLAUSIBLE_FROM_UNIX)
        .unwrap_or(OffsetDateTime::UNIX_EPOCH);
    let until = OffsetDateTime::now_utc() + time::Duration::days(1);
    t >= from && t <= until
}

/// A time hint when `raw` parses, otherwise a text hint carrying it unchanged.
pub fn time_hint(kind: HintKind, source: &str, raw: &str) -> Hint {
    match parse_iso8601(raw) {
        Some(p) => Hint::new(
            kind,
            source,
            raw,
            HintValue::Time {
                plausible: plausible(p.instant),
                utc: p.rfc3339,
            },
        ),
        None => Hint::text(kind, source, raw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_offset_forms() {
        for (raw, want) in [
            ("2026-05-14T10:02:00-0700", "2026-05-14T10:02:00-07:00"),
            ("2026-05-14T10:02:00-07:00", "2026-05-14T10:02:00-07:00"),
            ("2026-05-14T17:02:00Z", "2026-05-14T17:02:00Z"),
            (
                "2026-05-14T10:02:00.250+01:00",
                "2026-05-14T10:02:00.25+01:00",
            ),
            ("2026-05-14 10:02:00", "2026-05-14T10:02:00"),
            ("2026-05-14T10:02:00", "2026-05-14T10:02:00"),
            ("2026-05-14", "2026-05-14"),
        ] {
            let p = parse_iso8601(raw).unwrap_or_else(|| panic!("{raw}"));
            assert_eq!(p.rfc3339, want, "{raw}");
        }
        assert!(parse_iso8601("yesterday").is_none());
        assert!(parse_iso8601("2026-13-40").is_none());
    }

    #[test]
    fn non_ascii_offset_no_panic() {
        assert!(parse_iso8601("2026-05-14T10:02:00+0é0").is_none());
        assert!(parse_iso8601("2026-05-14T10:02:00+é").is_none());
        assert!(parse_iso8601("2026-05-1é").is_none());
    }

    #[test]
    fn parsers_never_panic_on_multibyte_input() {
        let base = [
            "2026-05-14T10:02:00.250+07:00",
            "+51.5072-000.1276+011.000/",
            "+3745.00-12225.00/",
        ];
        for b in base {
            for i in 0..=b.len() {
                for ins in ["é", "€", "𝄞", "\u{202e}"] {
                    let s = format!("{}{ins}{}", &b[..i], &b[i..]);
                    let _ = parse_iso8601(&s);
                    let _ = crate::iso6709::parse_iso6709(&s);
                }
            }
        }
    }

    #[test]
    fn plausibility_window() {
        assert!(!plausible(
            OffsetDateTime::from_unix_timestamp(86_400).unwrap()
        ));
        assert!(plausible(
            OffsetDateTime::from_unix_timestamp(1_778_000_000).unwrap()
        ));
        let far = OffsetDateTime::now_utc() + time::Duration::days(3);
        assert!(!plausible(far));
    }
}
