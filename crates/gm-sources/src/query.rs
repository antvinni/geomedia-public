//! Aircraft near an area during a time window, read straight from the day archive via its index.

use std::io::{Read, Seek, SeekFrom};

use gm_core::Epoch;

use crate::SourceError;
use crate::adsblol::{Day, DayIndex, MAX_TRACE_BYTES, MultiPart, TraceEntry};
use crate::area::Area;
use crate::trace::{AircraftTrack, parse_trace};

/// Aircraft whose traces come within this distance of the area are considered.
pub const PAD_M: f64 = 30_000.0;
/// Extra time read on each side of the window (≥ one ADS-B sample interval at cruise).
pub const EDGE_PAD_S: f64 = 60.0;

/// Index entries overlapping the padded area and the window [t0, t1].
pub fn matching<'a>(idx: &'a DayIndex, area: &Area, t0: Epoch, t1: Epoch) -> Vec<&'a TraceEntry> {
    let (s, w, n, e) = area.bbox();
    let pad_lat = PAD_M / 111_320.0;
    let pad_lon = PAD_M / (111_320.0 * (0.5 * (s + n)).to_radians().cos().max(1e-3));
    let (s, n) = (s - pad_lat, n + pad_lat);
    let (w, e) = (w - pad_lon, e + pad_lon);
    let (u0, u1) = (t0.ns as f64 * 1e-9, t1.ns as f64 * 1e-9);
    // Longitude intervals on [-180, 180], splitting boxes that wrap.
    let spans = |lo: f64, hi: f64| -> Vec<(f64, f64)> {
        if lo <= hi {
            vec![(lo, hi)]
        } else {
            vec![(lo, 180.0), (-180.0, hi)]
        }
    };
    let norm = |x: f64| ((x + 180.0).rem_euclid(360.0)) - 180.0;
    let area_spans = if e - w >= 360.0 {
        vec![(-180.0, 180.0)]
    } else {
        spans(norm(w), norm(e))
    };
    idx.entries
        .iter()
        .filter(|en| en.t1 >= u0 && en.t0 <= u1 && en.lat_max >= s && en.lat_min <= n)
        .filter(|en| {
            spans(en.lon_min, en.lon_max)
                .iter()
                .any(|&(a0, a1)| area_spans.iter().any(|&(b0, b1)| a1 >= b0 && a0 <= b1))
        })
        .collect()
}

pub fn query(
    idx: &DayIndex,
    day: &Day,
    area: &Area,
    t0: Epoch,
    t1: Epoch,
    origin: Epoch,
) -> Result<Vec<AircraftTrack>, SourceError> {
    query_reader(idx, MultiPart::open(&day.parts)?, area, t0, t1, origin)
}

pub fn query_reader<R: Read + Seek>(
    idx: &DayIndex,
    mut reader: R,
    area: &Area,
    t0: Epoch,
    t1: Epoch,
    origin: Epoch,
) -> Result<Vec<AircraftTrack>, SourceError> {
    let io = |source| SourceError::Io {
        path: "adsb archive".into(),
        source,
    };
    let mut out = Vec::new();
    for e in matching(idx, area, t0, t1) {
        if e.len > MAX_TRACE_BYTES {
            continue;
        }
        reader.seek(SeekFrom::Start(e.offset)).map_err(io)?;
        let mut data = vec![0u8; e.len as usize];
        reader.read_exact(&mut data).map_err(io)?;
        // Pad by one sample interval so tracks cover the window's edges.
        let (p0, p1) = (t0.plus_seconds(-EDGE_PAD_S), t1.plus_seconds(EDGE_PAD_S));
        let Ok(mut a) = parse_trace(&data, origin, Some((p0, p1))) else {
            continue;
        };
        // The day bbox only says the airframe was somewhere near at some time; keep tracks that
        // actually come within PAD_M of the area during the window.
        let (w0, w1) = (t0.seconds_since(origin), t1.seconds_since(origin));
        a.tracks.retain(|tr| {
            tr.knots.iter().any(|k| {
                let g = gm_core::ecef_to_geodetic(k.pos);
                k.t >= w0 - EDGE_PAD_S
                    && k.t <= w1 + EDGE_PAD_S
                    && area.near(g.lat_deg, g.lon_deg, PAD_M)
            })
        });
        if !a.tracks.is_empty() {
            out.push(a);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adsblol::index_parts;
    use crate::adsblol::tests::fixture_day;
    use std::cell::Cell;
    use std::rc::Rc;

    struct Counting<R> {
        inner: R,
        bytes: Rc<Cell<u64>>,
    }
    impl<R: Read> Read for Counting<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.bytes.set(self.bytes.get() + n as u64);
            Ok(n)
        }
    }
    impl<R: Seek> Seek for Counting<R> {
        fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(to)
        }
    }

    fn date() -> time::Date {
        time::Date::from_calendar_date(2026, time::Month::September, 8).unwrap()
    }
    const BASE: f64 = 1_788_825_600.0;

    #[test]
    fn query_reads_only_matches() {
        let dir = tempfile::tempdir().unwrap();
        let parts = fixture_day(dir.path(), 2049);
        let idx = index_parts(date(), &parts, dir.path()).unwrap();
        let area = Area::Circle {
            lat: 40.8765,
            lon: -73.525,
            radius_m: 10_000.0,
        };
        let (t0, t1) = (
            Epoch::from_unix_f64(BASE + 74_000.0),
            Epoch::from_unix_f64(BASE + 77_000.0),
        );
        let m = matching(&idx, &area, t0, t1);
        let names: Vec<&str> = m.iter().map(|e| e.icao.as_str()).collect();
        assert_eq!(names, ["a1dfeb", "ace391"]);
        let bytes = Rc::new(Cell::new(0));
        let reader = Counting {
            inner: MultiPart::open(&parts).unwrap(),
            bytes: bytes.clone(),
        };
        let got = query_reader(&idx, reader, &area, t0, t1, t0).unwrap();
        assert_eq!(got.len(), 2);
        let want: u64 = m.iter().map(|e| e.len).sum();
        assert_eq!(bytes.get(), want, "read only the matching entries");
    }

    #[test]
    fn query_time_window() {
        let dir = tempfile::tempdir().unwrap();
        let parts = fixture_day(dir.path(), 2049);
        let idx = index_parts(date(), &parts, dir.path()).unwrap();
        let area = Area::Circle {
            lat: 40.8765,
            lon: -73.525,
            radius_m: 10_000.0,
        };
        let (t0, t1) = (
            Epoch::from_unix_f64(BASE + 75_900.0),
            Epoch::from_unix_f64(BASE + 77_000.0),
        );
        let names: Vec<String> = matching(&idx, &area, t0, t1)
            .iter()
            .map(|e| e.icao.clone())
            .collect();
        assert_eq!(names, ["ace391"]);
        let london = Area::Circle {
            lat: 51.47,
            lon: -0.45,
            radius_m: 5000.0,
        };
        assert_eq!(
            matching(
                &idx,
                &london,
                Epoch::from_unix_f64(BASE),
                Epoch::from_unix_f64(BASE + 2000.0)
            )
            .len(),
            1
        );
    }

    #[test]
    fn query_keeps_only_aircraft_near_the_area_in_the_window() {
        use crate::adsblol::tests::{gz, trace_json};
        // One airframe flies London → New York the same day: its day bbox covers both, but at
        // 75 000 s it is over London, not New York.
        let dir = tempfile::tempdir().unwrap();
        let mut b = tar::Builder::new(Vec::new());
        let data = gz(&trace_json(
            "far001",
            BASE,
            &[
                (74_000.0, 51.47, -0.45),
                (76_000.0, 51.60, -1.0),
                (90_000.0, 40.80, -73.60),
            ],
        ));
        let mut h = tar::Header::new_gnu();
        h.set_size(data.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        b.append_data(&mut h, "./traces/01/trace_full_far001.json", &data[..])
            .unwrap();
        let p = dir.path().join("x.tar");
        std::fs::write(&p, b.into_inner().unwrap()).unwrap();
        let parts = vec![p];
        let idx = index_parts(date(), &parts, dir.path()).unwrap();
        let area = Area::Circle {
            lat: 40.8765,
            lon: -73.525,
            radius_m: 10_000.0,
        };
        let (t0, t1) = (
            Epoch::from_unix_f64(BASE + 74_900.0),
            Epoch::from_unix_f64(BASE + 75_100.0),
        );
        assert_eq!(
            matching(&idx, &area, t0, t1).len(),
            1,
            "the day bbox overlaps"
        );
        let got = query_reader(&idx, MultiPart::open(&parts).unwrap(), &area, t0, t1, t0).unwrap();
        assert!(
            got.is_empty(),
            "{:?}",
            got.iter().map(|a| &a.icao).collect::<Vec<_>>()
        );
    }

    #[test]
    fn query_tracks_cover_window_edges() {
        let dir = tempfile::tempdir().unwrap();
        let parts = fixture_day(dir.path(), 2049);
        let idx = index_parts(date(), &parts, dir.path()).unwrap();
        let area = Area::Circle {
            lat: 40.8765,
            lon: -73.525,
            radius_m: 10_000.0,
        };
        // Window edges fall between 20 s samples.
        let (t0, t1) = (
            Epoch::from_unix_f64(BASE + 75_031.0),
            Epoch::from_unix_f64(BASE + 75_169.0),
        );
        let got = query_reader(&idx, MultiPart::open(&parts).unwrap(), &area, t0, t1, t0).unwrap();
        let a = got.iter().find(|a| a.icao == "a1dfeb").unwrap();
        let tr = &a.tracks[0];
        assert!(
            tr.state_at(0.0).is_some() && tr.state_at(t1.seconds_since(t0)).is_some(),
            "{:?}",
            tr.span()
        );
    }
}
