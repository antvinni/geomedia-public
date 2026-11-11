//! `geomedia discover-bench`: how reliably free discovery finds aircraft Doppler curves, by class,
//! slant range and SNR, on simulated passes with known truth (no ADS-B, no location search).
//! A pass is found when an event's CPA time is within max(3 s, 2σ_tc) of the truth arrival and
//! its fitted speed and slant are within 2× of the truth; every other event is a false event.

use std::fmt::Write as _;

use anyhow::Result;
use gm_doppler::audio::{Audio, EventParams, discover};

/// One class of aircraft sound: rest frequency and relative harmonic amplitudes.
struct Class {
    name: &'static str,
    f0: f64,
    harmonics: &'static [f64],
    speed: f64,
}

const CLASSES: &[Class] = &[
    Class {
        name: "piston",
        f0: 80.0,
        harmonics: &[1.0, 0.6, 0.35, 0.2],
        speed: 60.0,
    },
    Class {
        name: "turboprop",
        f0: 103.0,
        harmonics: &[1.0, 0.5, 0.25],
        speed: 120.0,
    },
    Class {
        name: "helicopter",
        f0: 22.0,
        harmonics: &[1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
        speed: 50.0,
    },
    Class {
        name: "jet-low-tone",
        f0: 650.0,
        harmonics: &[1.0],
        speed: 80.0,
    },
    Class {
        name: "jet-high-tone",
        f0: 1800.0,
        harmonics: &[1.0],
        speed: 80.0,
    },
];

/// (altitude AGL, lateral CPA offset) giving slants of ~0.8, 1.5, 3 and 5 km.
const GEOMETRY: &[(f64, f64)] = &[
    (600.0, 500.0),
    (1000.0, 1100.0),
    (1500.0, 2600.0),
    (2000.0, 4600.0),
];
const SNR_DB: &[f64] = &[30.0, 22.0, 16.0, 10.0];

fn scenario(c: &Class, alt: f64, off: f64, snr: f64, seed: u64, amp: f64) -> String {
    let h: Vec<String> = c.harmonics.iter().map(|x| format!("{x:.2}")).collect();
    format!(
        "seed = {seed}\nsample_rate = 8000\nduration_s = 600.0\nstart_utc = \"2026-09-08T20:00:00Z\"\n\
         [observer]\nlat_deg = 40.8765\nlon_deg = -73.5250\nh_ell_m = -20.0\nmic_height_m = 1.5\n\
         [atmosphere]\nc = 343.0\nwind_enu = [0.0, 0.0, 0.0]\n[noise]\nsnr_db = {snr:.1}\n\
         [[aircraft]]\nid = \"a\"\nicao24 = \"bench1\"\ntype_code = \"{}\"\nf0 = {:.1}\nharmonics = [{}]\namplitude = {amp:.1}\n\
         [aircraft.straight]\nheading_deg = 80.0\nspeed_mps = {:.1}\nalt_agl_m = {alt:.1}\ncpa_offset_m = {off:.1}\ncpa_time_s = 300.0\nhalf_span_s = 280.0\n",
        c.name,
        c.f0,
        h.join(", "),
        c.speed
    )
}

/// (class, geometry, SNR) index and its outcome with the truth slant.
type CellResult = ((usize, usize, usize), Result<(Cell, f64)>);

/// One cell's outcome.
struct Cell {
    found: bool,
    false_events: usize,
    /// …of which from the template stage (they carry a z); the rest are ridge events.
    false_template: usize,
    tc_err: Option<f64>,
}

/// Recording conditions applied to every cell (Task 5: realism beyond white noise).
#[derive(Default)]
pub struct Conditions {
    /// Real background at 8 kHz, mixed under the pass at the cell's SNR (replacing white noise).
    pub background: Option<std::sync::Arc<Vec<f32>>>,
    /// ffmpeg round trip: "opus" (24 kb/s) or "aac" (64 kb/s).
    pub codec: Option<String>,
    /// Phone-style high-pass cut-off (Hz), 2nd-order Butterworth.
    pub highpass_hz: Option<f64>,
    /// Render every cell silent (amplitude 0): every event is false (or, on a real background, another
    /// source) — the false-event rate per hour under these conditions.
    pub no_pass: bool,
}

/// 2nd-order Butterworth high-pass (RBJ biquad), in place.
fn highpass(x: &mut [f32], rate: f64, fc: f64) {
    let w = std::f64::consts::TAU * fc / rate;
    let (cw, alpha) = (w.cos(), w.sin() / std::f64::consts::SQRT_2);
    let a0 = 1.0 + alpha;
    let (b0, b1, b2) = (
        (1.0 + cw) / 2.0 / a0,
        -(1.0 + cw) / a0,
        (1.0 + cw) / 2.0 / a0,
    );
    let (a1, a2) = (-2.0 * cw / a0, (1.0 - alpha) / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0, 0.0, 0.0);
    for v in x.iter_mut() {
        let x0 = f64::from(*v);
        let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        (x2, x1, y2, y1) = (x1, x0, y1, y0);
        *v = y0 as f32;
    }
}

/// ffmpeg round trip through `codec`; the decoded audio.
fn codec_round_trip(samples: &[f32], rate: u32, codec: &str, seed: u64) -> Result<Audio> {
    let dir = std::env::temp_dir();
    let tag = format!("gm-bench-{}-{seed}", std::process::id());
    let inp = dir.join(format!("{tag}-in.wav"));
    gm_sim::write_wav(&inp, rate, samples)?;
    let (args, ext): (&[&str], &str) = match codec {
        "opus" => (&["-c:a", "libopus", "-b:a", "24k"], "ogg"),
        "aac" => (&["-c:a", "aac", "-b:a", "64k"], "m4a"),
        c => anyhow::bail!("unknown codec {c:?}: use opus or aac"),
    };
    let out = dir.join(format!("{tag}-out.{ext}"));
    let st = std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&inp)
        .args(args)
        .arg(&out)
        .status()
        .map_err(|e| anyhow::anyhow!("ffmpeg not available: {e}"))?;
    let decoded = if st.success() {
        gm_doppler::audio::decode(&out).map_err(anyhow::Error::from)
    } else {
        Err(anyhow::anyhow!("ffmpeg failed for {codec}"))
    };
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&out);
    decoded
}

fn run_cell(
    c: &Class,
    alt: f64,
    off: f64,
    snr: f64,
    seed: u64,
    cond: &Conditions,
) -> Result<(Cell, f64)> {
    // With a real background the simulator's own noise is all but off; the background is
    // scaled to the white-noise level the cell's SNR implies (as gm-sim defines it).
    let sim_snr = if cond.background.is_some() { 80.0 } else { snr };
    let amp = if cond.no_pass { 0.0 } else { 1.0 };
    let s = gm_sim::parse(&scenario(c, alt, off, sim_snr, seed, amp))?;
    let r = gm_sim::render(&s)?;
    let t = &r.truth.events[0];
    let mut samples = r.samples;
    if let Some(bg) = &cond.background {
        let target = (f64::from(s.sample_rate) / 4.0 / 10f64.powf(snr / 10.0)).sqrt();
        let start = (seed as usize * 7919) % bg.len().max(1);
        let seg = |i: usize| bg[(start + i) % bg.len()];
        let rms = ((0..samples.len())
            .map(|i| f64::from(seg(i)).powi(2))
            .sum::<f64>()
            / samples.len() as f64)
            .sqrt()
            .max(1e-12);
        let k = (target / rms) as f32;
        for (i, v) in samples.iter_mut().enumerate() {
            *v += k * seg(i);
        }
    }
    if let Some(fc) = cond.highpass_hz {
        highpass(&mut samples, f64::from(s.sample_rate), fc);
    }
    let audio = match &cond.codec {
        Some(codec) => codec_round_trip(&samples, s.sample_rate, codec, seed)?,
        None => Audio::new(s.sample_rate, samples),
    };
    let ev = discover(&audio, &EventParams::default(), &[]);
    let near = |x: f64, y: f64| x > 0.0 && y > 0.0 && (x / y).ln().abs() <= 2f64.ln();
    let hit = ev.iter().position(|e| {
        (e.fit.tc - t.cpa_arrival_audio_s).abs() <= 3f64.max(2.0 * e.fit.sigma_tc)
            && near(e.fit.v, t.speed_mps)
            && near(e.fit.d, t.slant_m)
    });
    Ok((
        Cell {
            found: hit.is_some(),
            false_events: ev.len() - usize::from(hit.is_some()),
            false_template: ev
                .iter()
                .enumerate()
                .filter(|(i, e)| Some(*i) != hit && e.guide_z.is_some())
                .count(),
            tc_err: hit.map(|i| (ev[i].fit.tc - t.cpa_arrival_audio_s).abs()),
        },
        t.slant_m,
    ))
}

/// Which cells to run (all when empty): for iterating on a few cells in seconds.
#[derive(Default)]
pub struct Filter {
    pub class: Option<String>,
    pub slant_km: Option<f64>,
    pub snr: Option<f64>,
}

/// The geometry whose nominal slant (0.8, 1.5, 3, 5 km) is nearest `km`.
fn nearest_geometry(km: f64) -> usize {
    const NOMINAL: [f64; 4] = [0.8, 1.5, 3.0, 5.0];
    (0..NOMINAL.len())
        .min_by(|&a, &b| (NOMINAL[a] - km).abs().total_cmp(&(NOMINAL[b] - km).abs()))
        .unwrap_or(0)
}

pub fn run(filter: &Filter, cond: &Conditions) -> Result<()> {
    if let Some(c) = &filter.class
        && !CLASSES.iter().any(|x| x.name == c)
    {
        anyhow::bail!(
            "unknown class {c:?}: use one of {}",
            CLASSES
                .iter()
                .map(|x| x.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let want_c = |ci: usize| {
        filter
            .class
            .as_deref()
            .is_none_or(|c| CLASSES[ci].name == c)
    };
    let want_g = |gi: usize| filter.slant_km.is_none_or(|k| nearest_geometry(k) == gi);
    let want_s = |si: usize| filter.snr.is_none_or(|x| (SNR_DB[si] - x).abs() < 0.5);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# discovery benchmark (simulated passes, 600 s; found = tc within max(3 s, 2σ), v and d within 2×)\n\nconditions: {}{}{}\n",
        if cond.background.is_some() {
            "real background (other events may be real aircraft)"
        } else {
            "white noise"
        },
        cond.codec
            .as_deref()
            .map_or(String::new(), |c| format!(", {c} round trip")),
        cond.highpass_hz
            .map_or(String::new(), |f| format!(", {f:.0} Hz high-pass"))
    );
    let _ = write!(out, "| class | slant |");
    for snr in SNR_DB {
        let _ = write!(out, " {snr:.0} dB |");
    }
    let _ = writeln!(out, "\n|---|---|{}", "---|".repeat(SNR_DB.len()));
    // Every cell is independent: run them on all cores (discovery's own threads share them).
    let jobs: Vec<(usize, usize, usize)> = (0..CLASSES.len())
        .flat_map(|ci| {
            (0..GEOMETRY.len()).flat_map(move |gi| (0..SNR_DB.len()).map(move |si| (ci, gi, si)))
        })
        .filter(|&(ci, gi, si)| want_c(ci) && want_g(gi) && want_s(si))
        .collect();
    if jobs.is_empty() {
        anyhow::bail!("no cells match the filter");
    }
    let workers = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .div_ceil(2)
        .max(1);
    let chunk = jobs.len().div_ceil(workers);
    let results: Vec<CellResult> = std::thread::scope(|sc| {
        let hs: Vec<_> = jobs
            .chunks(chunk)
            .map(|part| {
                sc.spawn(move || {
                    part.iter()
                        .map(|&(ci, gi, si)| {
                            let (alt, off) = GEOMETRY[gi];
                            let seed = 1000 + (ci * 100 + gi * 10 + si) as u64;
                            (
                                (ci, gi, si),
                                run_cell(&CLASSES[ci], alt, off, SNR_DB[si], seed, cond),
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    let (mut found, mut total, mut false_ev, mut false_t) = (0, 0, 0, 0);
    for (ci, c) in CLASSES.iter().enumerate() {
        for gi in 0..GEOMETRY.len() {
            let mut row = String::new();
            let mut slant = 0.0;
            for si in 0..SNR_DB.len() {
                // Cells outside the filter were not run: shown as "·".
                let Some((_, res)) = results.iter().find(|(k, _)| *k == (ci, gi, si)) else {
                    row.push_str(" · |");
                    continue;
                };
                let (cell, s) = match res {
                    Ok(v) => v,
                    Err(e) => anyhow::bail!("{} cell failed: {e}", c.name),
                };
                slant = *s;
                total += 1;
                found += usize::from(cell.found);
                false_ev += cell.false_events;
                false_t += cell.false_template;
                let _ = write!(
                    row,
                    " {}{} |",
                    if cell.found {
                        format!("✓ {:.1} s", cell.tc_err.unwrap_or(0.0))
                    } else {
                        "✗".into()
                    },
                    if cell.false_events > 0 {
                        format!(" (+{})", cell.false_events)
                    } else {
                        String::new()
                    }
                );
            }
            if slant > 0.0 {
                let _ = writeln!(out, "| {} | {:.1} km |{row}", c.name, slant / 1000.0);
            }
        }
    }
    let _ = writeln!(
        out,
        "\nfound {found} of {total} passes; {false_ev} false events (≈{:.1} per hour; {false_t} from templates, {} from ridges)",
        false_ev as f64 / (total as f64 * 600.0 / 3600.0),
        false_ev - false_t
    );
    print!("{out}");
    Ok(())
}
