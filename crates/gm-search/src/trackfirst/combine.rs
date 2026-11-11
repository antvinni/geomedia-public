//! One answer from two: the blind search's and track-first's, by a fixed table (design doc,
//! "Combining the two answers"). Never a pin that one method contradicts.

use gm_core::{Geodetic, geodetic_to_ecef};
use serde::Serialize;

use crate::params::TF_AGREE_M;
use crate::search::{Outcome, Verdict};

/// Which method the combined answer comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Source {
    Blind,
    TrackFirst,
    Both,
    Neither,
}

fn dist(a: &Geodetic, b: &Geodetic) -> f64 {
    let (p, q) = (geodetic_to_ecef(*a), geodetic_to_ecef(*b));
    ((p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.z - q.z).powi(2)).sqrt()
}

fn ambiguous(v: &Verdict) -> bool {
    matches!(v, Verdict::AmbiguousMirror | Verdict::AmbiguousStream)
}

/// An outcome's candidate points: `best` and `mirror`.
fn points(o: &Outcome) -> Vec<Geodetic> {
    [o.best.as_ref(), o.mirror.as_ref()]
        .into_iter()
        .flatten()
        .map(|r| r.geo)
        .collect()
}

fn near(a: &Geodetic, pts: &[Geodetic]) -> bool {
    pts.iter().any(|p| dist(a, p) <= TF_AGREE_M)
}

fn disagree(base: &Outcome) -> Outcome {
    Outcome {
        verdict: Verdict::Inconclusive(
            "methods disagree: the blind search and track-first point to different places".into(),
        ),
        best: None,
        top: vec![],
        mirror: None,
        ..base.clone()
    }
}

/// The combined answer, its source, and one line saying why.
pub fn combine(blind: &Outcome, tf: Option<&Outcome>) -> (Outcome, Source, String) {
    let Some(tf) = tf else {
        return (
            blind.clone(),
            Source::Blind,
            "track-first did not run".into(),
        );
    };
    use Verdict::{Inconclusive, Located};
    match (&blind.verdict, &tf.verdict) {
        (Located, Located) => {
            let (b, t) = (blind.best.as_ref(), tf.best.as_ref());
            match (b, t) {
                (Some(b), Some(t)) if dist(&b.geo, &t.geo) <= TF_AGREE_M => (
                    tf.clone(),
                    Source::Both,
                    "both located, within 1 km: track-first's answer".into(),
                ),
                _ => (
                    disagree(blind),
                    Source::Neither,
                    "both located, far apart".into(),
                ),
            }
        }
        (Located, Inconclusive(_)) => (blind.clone(), Source::Blind, "blind located".into()),
        (Inconclusive(_), Located) => {
            (tf.clone(), Source::TrackFirst, "track-first located".into())
        }
        (v, Located) | (Located, v) if ambiguous(v) => {
            let (amb, loc, src) = if ambiguous(&blind.verdict) {
                (blind, tf, Source::TrackFirst)
            } else {
                (tf, blind, Source::Blind)
            };
            match loc.best.as_ref() {
                Some(b) if near(&b.geo, &points(amb)) => (
                    loc.clone(),
                    src,
                    "located on one of the other method's ambiguous candidates".into(),
                ),
                _ => (
                    disagree(blind),
                    Source::Neither,
                    "located away from the other method's candidates".into(),
                ),
            }
        }
        (a, b) if ambiguous(a) && ambiguous(b) => {
            let shared = points(blind).iter().any(|p| near(p, &points(tf)));
            if shared {
                (
                    blind.clone(),
                    Source::Both,
                    "both ambiguous, sharing a candidate".into(),
                )
            } else {
                (
                    disagree(blind),
                    Source::Neither,
                    "both ambiguous, no shared candidate".into(),
                )
            }
        }
        (a, Inconclusive(_)) if ambiguous(a) => {
            (blind.clone(), Source::Blind, "blind ambiguous".into())
        }
        (Inconclusive(_), b) if ambiguous(b) => (
            tf.clone(),
            Source::TrackFirst,
            "track-first ambiguous".into(),
        ),
        (Inconclusive(a), Inconclusive(b)) => (
            Outcome {
                verdict: Inconclusive(format!("{a}; track-first: {b}")),
                ..blind.clone()
            },
            Source::Neither,
            "neither located".into(),
        ),
        _ => (
            disagree(blind),
            Source::Neither,
            "unexpected verdict pair".into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Ranked;

    const LAT: f64 = 40.88;
    const LON: f64 = -73.52;
    /// Degrees of latitude per km.
    const KM: f64 = 1.0 / 111.2;

    fn at(dkm: f64) -> Ranked {
        Ranked {
            geo: Geodetic {
                lat_deg: LAT + dkm * KM,
                lon_deg: LON,
                h_m: 0.0,
            },
            east: 0.0,
            north: dkm * 1000.0,
            ll: 0.0,
            prob: 1.0,
        }
    }

    fn out(v: Verdict, best: Option<f64>, mirror: Option<f64>) -> Outcome {
        Outcome {
            verdict: v,
            best: best.map(at),
            top: vec![],
            mirror: mirror.map(at),
            delta_s: 0.0,
            matched: vec![],
            silent: vec![],
            region90_km2: 0.0,
            on_boundary: false,
            heat: vec![],
            frame_origin: at(0.0).geo,
            wind_enu: None,
            atmosphere: String::new(),
        }
    }

    fn inc() -> Outcome {
        out(Verdict::Inconclusive("x".into()), None, None)
    }
    fn loc(km: f64) -> Outcome {
        out(Verdict::Located, Some(km), None)
    }
    fn amb(a: f64, b: f64) -> Outcome {
        out(Verdict::AmbiguousMirror, Some(a), Some(b))
    }

    #[test]
    fn blind_only() {
        let (o, s, _) = combine(&loc(0.0), Some(&inc()));
        assert_eq!((o.verdict, s), (Verdict::Located, Source::Blind));
        assert_eq!(combine(&loc(0.0), None).1, Source::Blind);
    }

    #[test]
    fn tf_only() {
        let (o, s, _) = combine(&inc(), Some(&loc(2.0)));
        assert_eq!((o.verdict, s), (Verdict::Located, Source::TrackFirst));
        assert!((o.best.unwrap().north - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn both_agree_takes_track_first() {
        let (o, s, _) = combine(&loc(0.0), Some(&loc(0.9)));
        assert_eq!((o.verdict, s), (Verdict::Located, Source::Both));
        assert!((o.best.unwrap().north - 900.0).abs() < 1e-9);
    }

    #[test]
    fn both_disagree_is_inconclusive() {
        let (o, s, _) = combine(&loc(0.0), Some(&loc(1.1)));
        assert!(matches!(&o.verdict, Verdict::Inconclusive(w) if w.contains("methods disagree")));
        assert_eq!(s, Source::Neither);
        assert!(o.best.is_none());
    }

    #[test]
    fn ambiguous_and_inconclusive_stays_ambiguous() {
        let (o, s, _) = combine(&amb(0.0, 3.0), Some(&inc()));
        assert_eq!((o.verdict, s), (Verdict::AmbiguousMirror, Source::Blind));
        let t = out(Verdict::AmbiguousStream, Some(1.0), None);
        let (o, s, _) = combine(&inc(), Some(&t));
        assert_eq!(
            (o.verdict, s),
            (Verdict::AmbiguousStream, Source::TrackFirst)
        );
    }

    #[test]
    fn ambiguous_resolved_by_located() {
        let (o, s, _) = combine(&amb(0.0, 3.0), Some(&loc(3.5)));
        assert_eq!((o.verdict, s), (Verdict::Located, Source::TrackFirst));
        let (o, s, _) = combine(&loc(0.2), Some(&amb(0.0, 3.0)));
        assert_eq!((o.verdict, s), (Verdict::Located, Source::Blind));
    }

    #[test]
    fn ambiguous_contradicted() {
        let (o, s, _) = combine(&amb(0.0, 3.0), Some(&loc(1.6)));
        assert!(matches!(&o.verdict, Verdict::Inconclusive(w) if w.contains("methods disagree")));
        assert_eq!(s, Source::Neither);
    }

    #[test]
    fn neither() {
        let (o, s, _) = combine(&inc(), Some(&inc()));
        assert!(matches!(&o.verdict, Verdict::Inconclusive(w) if w.contains("track-first")));
        assert_eq!(s, Source::Neither);
    }
}
