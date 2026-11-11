//! ISO 6709 point strings as written by phones and cameras, e.g. "+51.5072-000.1276+011.000/".

/// Parse an ISO 6709 point into `(lat_deg, lon_deg, alt_m)`.
pub fn parse_iso6709(s: &str) -> Option<(f64, f64, Option<f64>)> {
    let s = s.trim().trim_end_matches('/');
    let parts = split_signed(s)?;
    let (lat, lon, alt) = match parts.as_slice() {
        [lat, lon] => (*lat, *lon, None),
        [lat, lon, alt] => (*lat, *lon, Some(*alt)),
        _ => return None,
    };
    let lat = angle(lat, 2)?;
    let lon = angle(lon, 3)?;
    let alt = match alt {
        Some(a) => Some(plain_number(a)?),
        None => None,
    };
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return None;
    }
    Some((lat, lon, alt))
}

/// Split "+12.3-045.6+7" into ["+12.3", "-045.6", "+7"]; every part must start with a sign.
fn split_signed(s: &str) -> Option<Vec<&str>> {
    if !s.starts_with(['+', '-']) {
        return None;
    }
    let mut parts = Vec::new();
    let mut start = 0;
    for (i, c) in s.char_indices().skip(1) {
        if c == '+' || c == '-' {
            parts.push(&s[start..i]);
            start = i;
        }
    }
    parts.push(&s[start..]);
    Some(parts)
}

/// A signed component in degrees, DDMM or DDMMSS form; `deg_digits` is 2 for latitude, 3 for longitude.
fn angle(part: &str, deg_digits: usize) -> Option<f64> {
    let (sign, body) = sign_and_body(part)?;
    let int_len = body.find('.').unwrap_or(body.len());
    let value = plain_number(body)?;
    let deg = if int_len <= deg_digits {
        value
    } else if int_len == deg_digits + 2 {
        let d = (value / 100.0).trunc();
        let m = value - d * 100.0;
        if m >= 60.0 {
            return None;
        }
        d + m / 60.0
    } else if int_len == deg_digits + 4 {
        let d = (value / 10_000.0).trunc();
        let m = ((value - d * 10_000.0) / 100.0).trunc();
        let sec = value - d * 10_000.0 - m * 100.0;
        if m >= 60.0 || sec >= 60.0 {
            return None;
        }
        d + m / 60.0 + sec / 3600.0
    } else {
        return None;
    };
    Some(sign * deg)
}

fn sign_and_body(part: &str) -> Option<(f64, &str)> {
    if let Some(b) = part.strip_prefix('+') {
        Some((1.0, b))
    } else {
        part.strip_prefix('-').map(|b| (-1.0, b))
    }
}

/// Digits with at most one decimal point, optionally signed.
fn plain_number(s: &str) -> Option<f64> {
    let body = s.strip_prefix(['+', '-']).unwrap_or(s);
    let ok = !body.is_empty()
        && body.chars().all(|c| c.is_ascii_digit() || c == '.')
        && body.matches('.').count() <= 1
        && body.chars().any(|c| c.is_ascii_digit());
    if ok { s.parse().ok() } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: (f64, f64, Option<f64>), want: (f64, f64, Option<f64>)) {
        assert!((got.0 - want.0).abs() < 1e-9, "lat {got:?} vs {want:?}");
        assert!((got.1 - want.1).abs() < 1e-9, "lon {got:?} vs {want:?}");
        match (got.2, want.2) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-9, "alt {got:?} vs {want:?}"),
            (None, None) => {}
            _ => panic!("alt {got:?} vs {want:?}"),
        }
    }

    #[test]
    fn parses_variants() {
        let lon_ddmm = -(122.0 + 25.0 / 60.0);
        let cases = [
            ("+51.5072-000.1276/", (51.5072, -0.1276, None)),
            (
                "+37.7749-122.4194+010.000/",
                (37.7749, -122.4194, Some(10.0)),
            ),
            ("+3745.00-12225.00/", (37.75, lon_ddmm, None)),
            ("+374500-1222500/", (37.75, lon_ddmm, None)),
            ("-33.8688+151.2093-005.5/", (-33.8688, 151.2093, Some(-5.5))),
            ("+51.5072-000.1276", (51.5072, -0.1276, None)),
        ];
        for (s, want) in cases {
            let got = parse_iso6709(s).unwrap_or_else(|| panic!("failed to parse {s}"));
            close(got, want);
        }
    }

    #[test]
    fn rejects_garbage() {
        for s in [
            "",
            "abc",
            "+91.0+000.0/",
            "+10.0+181.0/",
            "+51.5",
            "+51.5-0.12x/",
        ] {
            assert_eq!(parse_iso6709(s), None, "{s}");
        }
    }
}
