//! Calibration of the per-aircraft floor and cap on simulated noise only (never on a pass):
//! `measure_noise_z_quantiles` prints the numbers `TF_Z_FLOOR` / `TF_Z_CAP` were set from;
//! `noise_pairs_rarely_clear_the_floor` guards them on a render not used to set them.

#[cfg(test)]
mod tests {
    use gm_core::LocalFrame;
    use gm_doppler::audio::guide::{GuideImage, GuideParams, coarsen, guide_image_with};

    use crate::atmos::Atmosphere;
    use crate::grid::{AircraftClass, Candidate, Inputs, cell_at};
    use crate::guided::Band;
    use crate::params::{TF_CELL_M, TF_COARSE_BPO, TF_COARSE_HOP_S, TF_Z_CAP, TF_Z_FLOOR};
    use crate::testutil::{Fixture, decoys, fixture};
    use crate::trackfirst::maps::pair_map_with;
    use crate::trackfirst::maps::tests::area;

    /// `name` with every aircraft silent (noise only), optionally re-seeded.
    fn noise(name: &str, seed: Option<u64>) -> Fixture {
        fixture(name, move |s| {
            let s = s.replace("amplitude = 1.0", "amplitude = 0.0");
            match seed {
                Some(k) => s
                    .lines()
                    .map(|l| {
                        if l.starts_with("seed = ") {
                            format!("seed = {k}")
                        } else {
                            l.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                None => s,
            }
        })
    }

    /// Every pooled z of `aircraft` over a 3×3 grid of 1 km cells around the observer.
    fn all_z(f: &Fixture, img: &GuideImage, band: Band, aircraft: &[Candidate]) -> Vec<f64> {
        let a = area("40.8765,-73.5250~6km");
        let h = f.truth.observer.h_m - crate::params::MIC_H;
        let ground = move |_: f64, _: f64| h;
        let atmos = Atmosphere::still();
        let inp = Inputs {
            events: &[],
            aircraft,
            area: &a,
            when: f.when,
            duration_s: f.duration_s,
            ground: &ground,
            atmos: &atmos,
            noise: None,
            decoys: &[],
        };
        let frame = LocalFrame::new(f.truth.observer);
        let mut out = Vec::new();
        for ai in 0..aircraft.len() {
            for (ci, (e, n)) in (-1..=1)
                .flat_map(|i| (-1..=1).map(move |k| (i, k)))
                .enumerate()
            {
                let cell = cell_at(
                    &frame,
                    f64::from(e) * TF_CELL_M,
                    f64::from(n) * TF_CELL_M,
                    &ground,
                );
                if let Some(m) = pair_map_with(
                    img,
                    band,
                    &inp,
                    &frame,
                    (ai, 0, ci),
                    cell,
                    (-1e7, 1e7),
                    TF_CELL_M,
                    f64::NEG_INFINITY,
                ) {
                    out.extend(m.hits.iter().map(|h| f64::from(h.z)));
                }
            }
        }
        out
    }

    fn quantile(v: &mut [f64], q: f64) -> f64 {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    fn prop_noise_z(f: &Fixture) -> Vec<f64> {
        let img = coarsen(
            &guide_image_with(&f.audio, &GuideParams::prop()),
            TF_COARSE_HOP_S,
            TF_COARSE_BPO,
        );
        // The scenario's own (silent) aircraft plus 24 decoys of unlike geometry.
        let mut ac = f.aircraft.clone();
        ac.extend(decoys(f, 24));
        all_z(f, &img, Band::Prop, &ac)
    }

    #[test]
    #[ignore = "measurement: prints the numbers TF_Z_FLOOR and TF_Z_CAP were set from"]
    fn measure_noise_z_quantiles() {
        let mut z: Vec<f64> = Vec::new();
        for name in ["suite/quiet.toml", "suite/windy.toml"] {
            z.extend(prop_noise_z(&noise(name, None)));
        }
        let n = z.len();
        let q99 = quantile(&mut z, 0.99);
        let q999 = quantile(&mut z, 0.999);
        println!("prop: {n} pooled z; 99th {q99:.2}; 99.9th {q999:.2}");
        // Jets: their own image and band; decoys within the 3 km jet slant.
        let f = noise("faint_jet.toml", None);
        let img = coarsen(
            &guide_image_with(&f.audio, &GuideParams::jet()),
            TF_COARSE_HOP_S,
            TF_COARSE_BPO,
        );
        let mut jets: Vec<Candidate> = decoys(&f, 24)
            .into_iter()
            .map(|mut d| {
                d.class = AircraftClass::Jet;
                d
            })
            .collect();
        jets.extend(f.aircraft.iter().cloned().map(|mut a| {
            a.class = AircraftClass::Jet;
            a
        }));
        let mut zj = all_z(&f, &img, Band::Jet, &jets);
        let nj = zj.len();
        if nj > 0 {
            let j99 = quantile(&mut zj, 0.99);
            let j999 = quantile(&mut zj, 0.999);
            println!("jet: {nj} pooled z; 99th {j99:.2}; 99.9th {j999:.2}");
        } else {
            println!("jet: no pairs in range");
        }
    }

    #[test]
    fn noise_pairs_rarely_clear_the_floor() {
        // A render not used to set the floor: windy, re-seeded, silent.
        let z = prop_noise_z(&noise("suite/windy.toml", Some(9001)));
        assert!(z.len() > 1000, "{} values", z.len());
        let above = z.iter().filter(|&&x| x >= TF_Z_FLOOR).count() as f64 / z.len() as f64;
        assert!(above <= 0.01, "{:.3}% above the floor", 100.0 * above);
        let over_cap = z.iter().filter(|&&x| x >= TF_Z_FLOOR + TF_Z_CAP).count();
        assert!(
            over_cap as f64 <= 0.001 * z.len() as f64,
            "{over_cap} at the cap"
        );
    }
}
