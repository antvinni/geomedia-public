//! Place names → coordinates via OpenStreetMap Nominatim (cached, at most 1 request/s).

use crate::SourceError;
use crate::area::Geocoded;

/// The first result of a Nominatim `jsonv2` search response.
pub fn parse_nominatim(json: &str) -> Result<Geocoded, SourceError> {
    let v: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| SourceError::Invalid(format!("geocoder response: {e}")))?;
    let first = v
        .as_array()
        .and_then(|a| a.first())
        .ok_or_else(|| SourceError::Invalid("geocoder found no match".into()))?;
    let f = |k: &str| {
        first
            .get(k)
            .and_then(|x| x.as_str())
            .and_then(|x| x.parse::<f64>().ok())
    };
    let bb: Vec<f64> = first
        .get("boundingbox")
        .and_then(|b| b.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()?.parse().ok()).collect())
        .unwrap_or_default();
    let (Some(lat), Some(lon), [s, n, w, e]) = (f("lat"), f("lon"), bb.as_slice()) else {
        return Err(SourceError::Invalid(
            "geocoder response lacks coordinates".into(),
        ));
    };
    let display_name = first
        .get("display_name")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    Ok(Geocoded {
        display_name,
        lat,
        lon,
        bbox: (*s, *w, *n, *e),
    })
}

/// Geocode `query` (`country:XX` searches by ISO country code).
pub fn nominatim(query: &str) -> Result<Geocoded, SourceError> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    query.trim().to_lowercase().hash(&mut h);
    let cache = crate::cache_dir()
        .join("geocode")
        .join(format!("{:016x}.json", h.finish()));
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Ok(g) = parse_nominatim(&text)
    {
        return Ok(g);
    }
    let url = match query.trim().strip_prefix("country:") {
        Some(code) => format!(
            "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&featureType=country&country={}",
            encode(code.trim())
        ),
        None => format!(
            "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q={}",
            encode(query.trim())
        ),
    };
    throttle();
    let http = |reason: String| SourceError::Http {
        url: url.clone(),
        reason,
    };
    let resp = crate::cache::agent()
        .get(&url)
        .call()
        .map_err(|e| http(e.to_string()))?;
    if resp.status().as_u16() != 200 {
        return Err(http(format!("HTTP {}", resp.status().as_u16())));
    }
    let text = resp
        .into_body()
        .read_to_string()
        .map_err(|e| http(e.to_string()))?;
    let g = parse_nominatim(&text)?;
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(&cache, &text);
    }
    Ok(g)
}

/// Nominatim's usage policy: at most one request per second.
fn throttle() {
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let mut last = LAST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(t) = *last {
        let wait = std::time::Duration::from_millis(1100).saturating_sub(t.elapsed());
        std::thread::sleep(wait);
    }
    *last = Some(std::time::Instant::now());
}

fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            b => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_first_result() {
        let json = r#"[{"lat":"40.8653","lon":"-73.5318","display_name":"Oyster Bay, Town of Oyster Bay, Nassau County, New York, United States","boundingbox":["40.8478","40.8828","-73.5513","-73.5120"]},{"lat":"1","lon":"2","display_name":"other","boundingbox":["0","1","0","1"]}]"#;
        let g = parse_nominatim(json).unwrap();
        assert!((g.lat - 40.8653).abs() < 1e-9 && (g.lon + 73.5318).abs() < 1e-9);
        assert_eq!(g.bbox, (40.8478, -73.5513, 40.8828, -73.5120));
        assert!(g.display_name.starts_with("Oyster Bay"));
        let e = parse_nominatim("[]").unwrap_err().to_string();
        assert!(e.contains("no match"), "{e}");
    }
}
