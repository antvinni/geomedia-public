//! Rendering a scenario to samples plus ground truth.

use gm_core::{Epoch, Geodetic, LocalFrame, Vec3, ecef_to_geodetic, geodetic_to_ecef};
use gm_doppler::physics::cv::StraightLine;
use gm_doppler::physics::{Knot, Track};
use serde::{Deserialize, Serialize};

use crate::rng::SplitMix64;
use crate::scenario::{Scenario, SimError};

/// Seconds of raised-cosine fade at each end of a track.
const FADE_S: f64 = 2.0;
/// Output peak level.
const PEAK_DBFS: f64 = -1.0;

pub struct Rendered {
    pub samples: Vec<f32>,
    pub truth: Truth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Truth {
    pub observer: Geodetic,
    pub mic_ecef: Vec3,
    pub start_utc: String,
    pub sample_rate: u32,
    pub duration_s: f64,
    pub c: f64,
    pub wind_ecef: Vec3,
    pub events: Vec<TruthEvent>,
    pub tracks: Vec<TruthTrack>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthEvent {
    pub icao24: String,
    pub type_code: String,
    pub f0: f64,
    pub cpa_arrival_audio_s: f64,
    pub cpa_arrival_utc: String,
    pub slant_m: f64,
    pub speed_mps: f64,
    /// The aircraft has a broadband (jet-like) source.
    #[serde(default)]
    pub broadband: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthTrack {
    pub icao24: String,
    pub type_code: String,
    /// Knot times are seconds relative to `start_utc`.
    pub track: Track,
}

/// Emission time for receive time `t` by plain fixed-point iteration
/// τ ← t − |r − p(τ) − w(t − τ)|/c, with the path extended linearly past the track ends.
pub fn emission_fixed_point(
    track: &Track,
    obs: Vec3,
    t: f64,
    c: f64,
    wind: Vec3,
    warm: Option<f64>,
) -> f64 {
    fixed_point(track, obs, t, c, wind, warm).0
}

/// The fixed-point iteration and the number of updates it took.
pub(crate) fn fixed_point(
    track: &Track,
    obs: Vec3,
    t: f64,
    c: f64,
    wind: Vec3,
    warm: Option<f64>,
) -> (f64, u32) {
    let mut tau = warm.unwrap_or(t);
    let tol = (4.0 * f64::EPSILON * t.abs()).max(1e-12);
    for i in 0..200 {
        let next = t - (obs - extended(track, tau) - wind * (t - tau)).norm() / c;
        let done = (next - tau).abs() <= tol;
        tau = next;
        if done {
            return (tau, i);
        }
    }
    (tau, 200)
}

/// Track position, continued linearly along the end tangents outside the span.
fn extended(track: &Track, tau: f64) -> Vec3 {
    if let Some((p, _)) = track.state_at(tau) {
        return p;
    }
    let (Some(first), Some(last)) = (track.knots.first(), track.knots.last()) else {
        return Vec3::ZERO;
    };
    if tau < first.t {
        first.pos + first.vel * (tau - first.t)
    } else {
        last.pos + last.vel * (tau - last.t)
    }
}

/// A straight ground track with constant along-track acceleration and vertical rate; at τ = 0
/// (the geometric CPA time) position and speed are those of `line`.
fn manoeuvre_track(line: &StraightLine, accel: f64, climb: f64, id: &str) -> Track {
    let f = LocalFrame::new(line.frame_origin);
    let h = line.heading_deg.to_radians();
    let dir = Vec3::new(h.sin(), h.cos(), 0.0);
    let right = Vec3::new(h.cos(), -h.sin(), 0.0);
    let up = Vec3::new(0.0, 0.0, 1.0);
    let at_cpa = right * -line.cpa_offset_m + up * line.alt_m;
    let n = (2.0 * line.half_span_s / line.dt_s).round().max(1.0) as usize;
    let knots = (0..=n)
        .map(|i| {
            let tau = -line.half_span_s + i as f64 * line.dt_s;
            let along = line.speed_mps * tau + 0.5 * accel * tau * tau;
            Knot {
                t: line.cpa_time_s + tau,
                pos: f.from_enu(at_cpa + dir * along + up * (climb * tau)),
                vel: f.vec_from_enu(dir * (line.speed_mps + accel * tau) + up * climb),
            }
        })
        .collect();
    Track {
        id: id.to_string(),
        knots,
    }
}

fn envelope(tau: f64, t0: f64, t1: f64) -> f64 {
    if tau < t0 || tau > t1 {
        return 0.0;
    }
    let x = ((tau - t0).min(t1 - tau) / FADE_S).min(1.0);
    0.5 - 0.5 * (std::f64::consts::PI * x).cos()
}

/// A broadband source: seeded noise in octave bands (a broad hump, −3 dB per octave either side
/// of the band's geometric centre), stored on the source's own clock and read at emission time,
/// so the received spectrum is Doppler-scaled; per-band ISO 9613-1 absorption over the path.
struct Broadband {
    tau0: f64,
    rate: f64,
    /// Per octave band: (centre Hz, samples at 1 km). One summed band when absorption is off.
    bands: Vec<(f64, Vec<f32>)>,
    absorb: bool,
}

impl Broadband {
    fn new(
        b: &crate::scenario::BroadbandCfg,
        id: &str,
        seed: u64,
        sr: f64,
        noise_sigma: f64,
        span: (f64, f64),
    ) -> Broadband {
        let n = ((span.1 - span.0) * sr).ceil() as usize + 2;
        let peak = (b.low_hz * b.high_hz).sqrt();
        let centres: Vec<f64> = (0..12)
            .map(|k| 62.5 * 2f64.powi(k))
            .filter(|&fc| fc >= b.low_hz && fc <= b.high_hz && fc * 2f64.sqrt() < sr / 2.0)
            .collect();
        let tag = id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, c| {
            (h ^ u64::from(c)).wrapping_mul(0x0100_0000_01b3)
        });
        let mut rng = SplitMix64::new(seed ^ tag);
        // Total variance at 1 km: the noise floor's density × 10^(level/10) × bandwidth.
        let noise_psd = noise_sigma * noise_sigma / (sr / 2.0);
        let total_var = noise_psd * 10f64.powf(b.level_db / 10.0) * (b.high_hz - b.low_hz);
        let weights: Vec<f64> = centres
            .iter()
            .map(|fc| 10f64.powf(-0.3 * (fc / peak).log2().abs()))
            .collect();
        let wsum: f64 = weights.iter().sum::<f64>().max(1e-12);
        let mut bands: Vec<(f64, Vec<f32>)> = centres
            .iter()
            .zip(&weights)
            .map(|(&fc, &w)| {
                let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
                let y = bandpass(&x, fc, sr);
                let var = y.iter().map(|v| v * v).sum::<f64>() / n as f64;
                let g = (total_var * w / wsum / var.max(1e-30)).sqrt();
                (fc, y.iter().map(|v| (v * g) as f32).collect())
            })
            .collect();
        if !b.absorption && bands.len() > 1 {
            let mut sum = vec![0f32; n];
            for (_, y) in &bands {
                for (s, v) in sum.iter_mut().zip(y) {
                    *s += v;
                }
            }
            bands = vec![(peak, sum)];
        }
        Broadband {
            tau0: span.0,
            rate: sr,
            bands,
            absorb: b.absorption,
        }
    }

    /// The source's value at emission time `tau` (at 1 km), with absorption over `r` metres.
    fn at(&self, tau: f64, r: f64) -> f64 {
        let x = (tau - self.tau0) * self.rate;
        if x < 0.0 {
            return 0.0;
        }
        let i = x.floor() as usize;
        let frac = x - i as f64;
        self.bands
            .iter()
            .map(|(fc, y)| {
                if i + 1 >= y.len() {
                    return 0.0;
                }
                let v = f64::from(y[i]) * (1.0 - frac) + f64::from(y[i + 1]) * frac;
                if self.absorb {
                    let a = gm_doppler::physics::absorption::alpha_db_per_m(
                        *fc,
                        gm_doppler::physics::absorption::STD_TEMP_C,
                        gm_doppler::physics::absorption::STD_RH_PCT,
                        gm_doppler::physics::absorption::STD_P_KPA,
                    );
                    v * 10f64.powf(-a * (r - 1000.0) / 20.0)
                } else {
                    v
                }
            })
            .sum()
    }
}

/// Octave band-pass (RBJ biquad, constant 0 dB peak gain, Q = √2), applied twice.
fn bandpass(x: &[f64], fc: f64, sr: f64) -> Vec<f64> {
    let w0 = std::f64::consts::TAU * fc / sr;
    let alpha = w0.sin() / (2.0 * std::f64::consts::SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b2) = (alpha / a0, -alpha / a0);
    let (a1, a2) = (-2.0 * w0.cos() / a0, (1.0 - alpha) / a0);
    let pass = |x: &[f64]| -> Vec<f64> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        x.iter()
            .map(|&v| {
                let y = b0 * v + b2 * x2 - a1 * y1 - a2 * y2;
                x2 = x1;
                x1 = v;
                y2 = y1;
                y1 = y;
                y
            })
            .collect()
    };
    pass(&pass(x))
}

pub fn render(s: &Scenario) -> Result<Rendered, SimError> {
    let start = Epoch::from_rfc3339(&s.start_utc).ok_or_else(|| SimError::Invalid {
        field: "start_utc".into(),
        reason: "must be RFC 3339".into(),
    })?;
    let mic = Geodetic {
        lat_deg: s.observer.lat_deg,
        lon_deg: s.observer.lon_deg,
        h_m: s.observer.h_ell_m + s.observer.mic_height_m,
    };
    let frame = LocalFrame::new(mic);
    let obs = geodetic_to_ecef(mic);
    let c = s.atmosphere.c;
    let [we, wn, wu] = s.atmosphere.wind_enu;
    let wind = frame.vec_from_enu(Vec3::new(we, wn, wu));
    let sr = f64::from(s.sample_rate);
    let n = (s.duration_s * sr).round() as usize;
    let mut y = vec![0f64; n];
    let mut events = Vec::new();
    let mut tracks = Vec::new();
    // White noise: an amplitude-1 tone (power 1/2) sits snr_db above the noise power in 1 Hz.
    let sigma = (sr / 4.0 / 10f64.powf(s.noise.snr_db / 10.0)).sqrt();
    for a in &s.aircraft {
        let st = &a.straight;
        let line = StraightLine {
            frame_origin: mic,
            heading_deg: st.heading_deg,
            speed_mps: st.speed_mps,
            alt_m: st.alt_agl_m - s.observer.mic_height_m,
            cpa_offset_m: st.cpa_offset_m,
            cpa_time_s: st.cpa_time_s,
            half_span_s: st.half_span_s,
            dt_s: s.adsb_dt_s,
        };
        let track = if st.accel_mps2 == 0.0 && st.climb_mps == 0.0 {
            line.track(&a.icao24)
        } else {
            manoeuvre_track(&line, st.accel_mps2, st.climb_mps, &a.icao24)
        };
        let (t0, t1) = track.span();
        let bb = a
            .broadband
            .as_ref()
            .filter(|_| a.audible)
            .map(|b| Broadband::new(b, &a.id, s.seed, sr, sigma, (t0 - 1.0, t1 + 1.0)));
        let mut prev: Option<(f64, f64, f64)> = None; // (t, τ, dτ/dt)
        for (i, out) in y.iter_mut().enumerate().filter(|_| a.audible) {
            let t = i as f64 / sr;
            let warm = prev.map(|(tp, taup, rate)| taup + (t - tp) * rate);
            let tau = emission_fixed_point(&track, obs, t, c, wind, warm);
            if let Some((tp, taup, _)) = prev {
                prev = Some((t, tau, (tau - taup) / (t - tp)));
            } else {
                prev = Some((t, tau, 1.0));
            }
            let env = envelope(tau, t0, t1);
            if env == 0.0 {
                continue;
            }
            let r = (c * (t - tau)).max(1.0);
            let phase = a.f0 * (tau + 0.5 * a.drift_per_s * tau * tau);
            let tone: f64 = a
                .harmonics
                .iter()
                .enumerate()
                .map(|(k, ak)| ak * (std::f64::consts::TAU * (k as f64 + 1.0) * phase).sin())
                .sum();
            *out += a.amplitude * 1000.0 / r * env * tone;
            if let Some(b) = &bb {
                *out += 1000.0 / r * env * b.at(tau, r);
            }
        }
        events.push(truth_event(a, &track, obs, c, wind, start));
        tracks.push(TruthTrack {
            icao24: a.icao24.clone(),
            type_code: a.type_code.clone(),
            track,
        });
    }
    let mut rng = SplitMix64::new(s.seed);
    for v in &mut y {
        *v += sigma * rng.normal();
    }
    let peak = y.iter().fold(0f64, |m, v| m.max(v.abs()));
    let gain = if peak > 0.0 {
        10f64.powf(PEAK_DBFS / 20.0) / peak
    } else {
        1.0
    };
    let samples = y.iter().map(|v| (v * gain) as f32).collect();
    let truth = Truth {
        observer: Geodetic {
            h_m: s.observer.h_ell_m,
            ..mic
        },
        mic_ecef: obs,
        start_utc: start.to_rfc3339(),
        sample_rate: s.sample_rate,
        duration_s: s.duration_s,
        c,
        wind_ecef: wind,
        events,
        tracks,
    };
    Ok(Rendered { samples, truth })
}

/// Truth CPA by a direct 1 ms scan (independent of the physics crate's CPA solver).
fn truth_event(
    a: &crate::scenario::AircraftCfg,
    track: &Track,
    obs: Vec3,
    c: f64,
    wind: Vec3,
    start: Epoch,
) -> TruthEvent {
    let (t0, t1) = track.span();
    let steps = ((t1 - t0) / 1e-3) as usize;
    let mut best = (t0, f64::INFINITY);
    for i in 0..=steps {
        let tau = t0 + i as f64 * 1e-3;
        if let Some((p, _)) = track.state_at(tau) {
            let d = (p - obs).norm();
            if d < best.1 {
                best = (tau, d);
            }
        }
    }
    let (t_emit, slant) = best;
    let (p, v) = track.state_at(t_emit).unwrap_or((obs, Vec3::ZERO));
    let q = obs - p;
    let aa = c * c - wind.dot(wind);
    let qw = q.dot(wind);
    let delay = (-qw + (qw * qw + aa * q.dot(q)).sqrt()) / aa;
    let arrival = t_emit + delay;
    let g = ecef_to_geodetic(p);
    let (la, lo) = (g.lat_deg.to_radians(), g.lon_deg.to_radians());
    let up = Vec3::new(la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin());
    TruthEvent {
        icao24: a.icao24.clone(),
        type_code: a.type_code.clone(),
        f0: a.f0,
        cpa_arrival_audio_s: arrival,
        cpa_arrival_utc: start.plus_seconds(arrival).to_rfc3339(),
        slant_m: slant,
        speed_mps: (v - up * v.dot(up)).norm(),
        broadband: a.broadband.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{parse, tests::SMALL};
    use gm_doppler::physics::{Medium, cpa, doppler_factor};

    /// One jet-like aircraft: silent tones, a broadband source.
    fn jet(absorption: bool, low_hz: f64, high_hz: f64) -> crate::scenario::Scenario {
        parse(&format!(
            r#"
seed = 7
sample_rate = 8000
duration_s = 120.0
start_utc = "2026-09-08T20:10:00Z"
[observer]
lat_deg = 40.88
lon_deg = -73.52
h_ell_m = 100.0
mic_height_m = 1.5
[atmosphere]
c = 343.0
wind_enu = [0.0, 0.0, 0.0]
[noise]
snr_db = 0.0
[[aircraft]]
id = "jet"
icao24 = "a1b2c3"
type_code = "B738"
f0 = 100.0
harmonics = [0.0]
amplitude = 1.0
[aircraft.broadband]
level_db = 40.0
low_hz = {low_hz}
high_hz = {high_hz}
absorption = {absorption}
[aircraft.straight]
heading_deg = 45.0
speed_mps = 120.0
alt_agl_m = 1200.0
cpa_offset_m = 1500.0
cpa_time_s = 60.0
half_span_s = 70.0
"#
        ))
        .unwrap()
    }

    /// Mean power per `hop_s` block, smoothed over `smooth_s`.
    fn power_envelope(x: &[f32], sr: f64, hop_s: f64, smooth_s: f64) -> Vec<f64> {
        let h = (hop_s * sr) as usize;
        let p: Vec<f64> = x
            .chunks(h)
            .map(|c| c.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / c.len() as f64)
            .collect();
        let k = ((smooth_s / hop_s) as usize / 2).max(1);
        (0..p.len())
            .map(|i| {
                let (a, b) = (i.saturating_sub(k), (i + k + 1).min(p.len()));
                p[a..b].iter().sum::<f64>() / (b - a) as f64
            })
            .collect()
    }

    #[test]
    fn silent_aircraft_have_tracks_but_no_audio() {
        let loud = render(&jet(true, 100.0, 3000.0)).unwrap();
        let mut sc = jet(true, 100.0, 3000.0);
        sc.aircraft[0].audible = false;
        let quiet = render(&sc).unwrap();
        assert_eq!(quiet.truth.tracks.len(), 1);
        let p = |x: &[f32]| x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
        // Only the noise floor (normalised to the same peak) is left: no hump in the middle.
        let n = quiet.samples.len();
        let mid = p(&quiet.samples[n / 2 - 8000..n / 2 + 8000]);
        let start = p(&quiet.samples[..16000]);
        assert!(mid < 3.0 * start, "{mid} vs {start}");
        assert!(p(&loud.samples[n / 2 - 8000..n / 2 + 8000]) > 10.0 * p(&loud.samples[..16000]));
    }

    #[test]
    fn broadband_pass_has_a_loudness_hump_at_cpa() {
        let r = render(&jet(true, 100.0, 3000.0)).unwrap();
        assert!(r.truth.events[0].broadband);
        let env = power_envelope(&r.samples, 8000.0, 0.5, 5.0);
        let peak = env
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| (i as f64 + 0.5) * 0.5)
            .unwrap();
        let arrival = r.truth.events[0].cpa_arrival_audio_s;
        assert!(
            (peak - arrival).abs() <= 2.0,
            "peak {peak} s vs arrival {arrival} s"
        );
        // A hump, not a plateau: well above the start of the recording.
        assert!(
            env[env.len() / 2] > 10.0 * env[4],
            "{} vs {}",
            env[env.len() / 2],
            env[4]
        );
    }

    #[test]
    fn broadband_spectrum_is_doppler_scaled() {
        // One octave (500 Hz band), so the shifted spectrum stays inside the analysed range.
        let s = jet(false, 300.0, 900.0);
        let r = render(&s).unwrap();
        let a = gm_doppler::audio::Audio::new(8000, r.samples.clone());
        let sp = gm_doppler::audio::spectrogram(
            &a,
            &gm_doppler::audio::SpecParams {
                window_s: 1.0,
                hop_s: 0.5,
                zero_pad: 1,
                f_min: 150.0,
                f_max: 2500.0,
            },
        );
        let centroid = |t: f64| -> f64 {
            let (mut num, mut den) = (0.0, 0.0);
            for (i, &ti) in sp.times.iter().enumerate() {
                if (ti - t).abs() <= 2.0 {
                    for (j, &f) in sp.freqs.iter().enumerate() {
                        let p = 10f64.powf(f64::from(sp.power_db[i][j]) / 10.0);
                        num += p * f;
                        den += p;
                    }
                }
            }
            num / den
        };
        let arrival = r.truth.events[0].cpa_arrival_audio_s;
        let (t1, t2) = (arrival - 15.0, arrival + 15.0);
        let tr = &r.truth.tracks[0].track;
        let m = Medium::still(343.0).unwrap();
        let k1 = doppler_factor(tr, r.truth.mic_ecef, t1, m).unwrap();
        let k2 = doppler_factor(tr, r.truth.mic_ecef, t2, m).unwrap();
        let got = centroid(t1) / centroid(t2);
        let want = k1 / k2;
        assert!(
            (got / want - 1.0).abs() < 0.10,
            "centroid ratio {got} vs Doppler {want}"
        );
    }

    fn fixture(name: &str) -> crate::scenario::Scenario {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sim")
            .join(name);
        crate::scenario::load(&p).unwrap()
    }

    #[test]
    fn straight_unchanged_when_no_manoeuvre() {
        // Values rendered from two_props.toml before manoeuvres existed.
        let r = render(&fixture("two_props.toml")).unwrap();
        assert_eq!(r.samples[1000].to_bits(), 0xbe83_9420);
        assert_eq!(r.samples[400_000], 0.074_344_41);
        let sum: f64 = r.samples.iter().map(|x| f64::from(x.abs())).sum();
        assert!((sum - 1_091_962.186_017_358).abs() < 1e-6, "{sum}");
    }

    #[test]
    fn manoeuvre_track_kinematics() {
        let s = fixture("decel.toml");
        let r = render(&s).unwrap();
        let b = &s.aircraft[1];
        let tr = &r.truth.tracks[1].track;
        let mic = r.truth.observer;
        let frame = LocalFrame::new(Geodetic {
            h_m: mic.h_m + s.observer.mic_height_m,
            ..mic
        });
        for tau in [-60.0, 60.0] {
            let (p, v) = tr.state_at(b.straight.cpa_time_s + tau).unwrap();
            let ve = frame.vec_to_enu(v);
            let speed = ve.x.hypot(ve.y);
            assert!((speed - (95.0 - 0.6 * tau)).abs() < 1e-6, "{tau}: {speed}");
            assert!((ve.z - -6.0).abs() < 1e-6, "{tau}: {}", ve.z);
            let up = frame.to_enu(p).z;
            let want = 1200.0 - 6.0 * tau - s.observer.mic_height_m;
            assert!((up - want).abs() < 1e-3, "{tau}: {up} vs {want}");
        }
    }

    #[test]
    fn deterministic() {
        let s = parse(SMALL).unwrap();
        let a = render(&s).unwrap();
        let b = render(&s).unwrap();
        assert_eq!(a.samples.len(), 240_000);
        assert!(
            a.samples
                .iter()
                .zip(&b.samples)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        let peak = a.samples.iter().fold(0f32, |m, x| m.max(x.abs()));
        assert!((peak - 10f32.powf(-1.0 / 20.0)).abs() < 1e-4, "{peak}");
    }

    #[test]
    fn sim_dtau_matches_physics() {
        let s = parse(SMALL).unwrap();
        let r = render(&s).unwrap();
        let tr = &r.truth.tracks[0].track;
        let m = Medium::new(r.truth.c, r.truth.wind_ecef).unwrap();
        let wind = r.truth.wind_ecef;
        for i in 0..400 {
            let t = 5.0 + f64::from(i) * 0.1;
            let h = 1e-3;
            let a = emission_fixed_point(tr, r.truth.mic_ecef, t - h, r.truth.c, wind, None);
            let b = emission_fixed_point(tr, r.truth.mic_ecef, t + h, r.truth.c, wind, None);
            let fd = (b - a) / (2.0 * h);
            let f = doppler_factor(tr, r.truth.mic_ecef, t, m).unwrap();
            assert!((fd - f).abs() < 1e-6, "t={t} {fd} {f}");
        }
    }

    #[test]
    fn truth_cpa_matches_physics() {
        let s = parse(SMALL).unwrap();
        let r = render(&s).unwrap();
        let m = Medium::new(r.truth.c, r.truth.wind_ecef).unwrap();
        let ev = &r.truth.events[0];
        let c = cpa(&r.truth.tracks[0].track, r.truth.mic_ecef, m).unwrap();
        assert!(
            (c.t_arrive - ev.cpa_arrival_audio_s).abs() < 1e-3,
            "{c:?} {ev:?}"
        );
        assert!((c.slant_m - ev.slant_m).abs() < 0.5, "{c:?} {ev:?}");
        assert!((ev.speed_mps - 130.0).abs() < 0.1);
        assert_eq!(ev.cpa_arrival_utc.get(..16), Some("2026-09-08T20:50"));
    }

    #[test]
    fn fixed_point_converges_at_large_t() {
        let s = parse(
            &SMALL
                .replace("cpa_time_s = 25.0", "cpa_time_s = 19000.0")
                .replace("half_span_s = 80.0", "half_span_s = 200.0"),
        )
        .unwrap();
        let mic = gm_core::Geodetic {
            lat_deg: s.observer.lat_deg,
            lon_deg: s.observer.lon_deg,
            h_m: 0.0,
        };
        let tr = StraightLine {
            frame_origin: mic,
            heading_deg: 70.0,
            speed_mps: 130.0,
            alt_m: 1500.0,
            cpa_offset_m: 900.0,
            cpa_time_s: 19000.0,
            half_span_s: 200.0,
            dt_s: 1.0,
        }
        .track("a");
        let obs = geodetic_to_ecef(mic);
        let worst = (0..200)
            .map(|i| {
                fixed_point(
                    &tr,
                    obs,
                    18950.0 + f64::from(i) * 0.731,
                    343.0,
                    Vec3::ZERO,
                    None,
                )
                .1
            })
            .max()
            .unwrap();
        assert!(worst < 40, "{worst}");
    }

    #[test]
    fn wav_header() {
        let s = parse(SMALL).unwrap();
        let r = render(&s).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.wav");
        crate::wav::write_wav(&p, s.sample_rate, &r.samples).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(bytes.len(), 44 + r.samples.len() * 2);
        assert_eq!(gm_media::sniff(&bytes[..64]), gm_media::Format::Wav);
    }
}
