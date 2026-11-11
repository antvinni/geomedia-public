//! Persistent narrowband lines (hum, engines, fans): bins that stand out from their spectral
//! neighbours through most of the recording. They can't be a passing aircraft (a pass sweeps),
//! and left in they take the peak slots a faint pass needs.

use super::spectrogram::Spectrogram;

/// A line must be this strong (whitened, mean over a quarter of the recording)…
pub const LINE_DB: f32 = 6.0;
/// …in at least this many of the recording's four quarters.
pub const LINE_QUARTERS: usize = 3;
/// Shorter recordings get no mask (too little time to tell a line from a slow pass).
pub const MIN_RECORDING_S: f64 = 30.0;

/// One flag per bin: a persistent line, widened by one bin each side.
pub fn persistent_lines(s: &Spectrogram) -> Vec<bool> {
    let nb = s.freqs.len();
    let nf = s.whitened.len();
    let span = match (s.times.first(), s.times.last()) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    };
    if nf < 4 || span < MIN_RECORDING_S {
        return vec![false; nb];
    }
    let q = nf / 4;
    let core: Vec<bool> = (0..nb)
        .map(|b| {
            (0..4)
                .filter(|k| {
                    let rows = &s.whitened[k * q..if *k == 3 { nf } else { (k + 1) * q }];
                    let mean = rows.iter().map(|fr| fr[b]).sum::<f32>() / rows.len() as f32;
                    mean >= LINE_DB
                })
                .count()
                >= LINE_QUARTERS
        })
        .collect();
    (0..nb)
        .map(|b| core[b] || (b > 0 && core[b - 1]) || (b + 1 < nb && core[b + 1]))
        .collect()
}
