//! Atmospheric absorption of sound (ISO 9613-1): pure-tone attenuation coefficient.

/// Standard conditions used when no weather is known.
pub const STD_TEMP_C: f64 = 15.0;
pub const STD_RH_PCT: f64 = 70.0;
pub const STD_P_KPA: f64 = 101.325;

/// Pure-tone attenuation coefficient α (dB per metre) at `f_hz`, air temperature `temp_c`,
/// relative humidity `rh_pct` and pressure `p_kpa` (ISO 9613-1, equations 3–5 and B.1).
pub fn alpha_db_per_m(f_hz: f64, temp_c: f64, rh_pct: f64, p_kpa: f64) -> f64 {
    const T0: f64 = 293.15; // reference temperature (K)
    const T01: f64 = 273.16; // triple-point isotherm (K)
    const PR: f64 = 101.325; // reference pressure (kPa)
    let t = temp_c + 273.15;
    let pa = p_kpa / PR;
    // Molar concentration of water vapour (%), from relative humidity.
    let c = -6.8346 * (T01 / t).powf(1.261) + 4.6151;
    let h = rh_pct * 10f64.powf(c) / pa;
    let tr = t / T0;
    let fr_o = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    let fr_n =
        pa * tr.powf(-0.5) * (9.0 + 280.0 * h * (-4.170 * (tr.powf(-1.0 / 3.0) - 1.0)).exp());
    let f2 = f_hz * f_hz;
    8.686
        * f2
        * (1.84e-11 / pa * tr.sqrt()
            + tr.powf(-2.5)
                * (0.01275 * (-2239.1 / t).exp() / (fr_o + f2 / fr_o)
                    + 0.1068 * (-3352.0 / t).exp() / (fr_n + f2 / fr_n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ISO 9613-2 Table 2 (from ISO 9613-1), 10 °C and 70 % relative humidity, dB/km.
    #[test]
    fn alpha_matches_iso9613_values() {
        for (f, want) in [
            (250.0, 1.0),
            (500.0, 1.9),
            (1000.0, 3.7),
            (2000.0, 9.7),
            (4000.0, 32.8),
            (8000.0, 117.0),
        ] {
            let got = alpha_db_per_m(f, 10.0, 70.0, STD_P_KPA) * 1000.0;
            assert!(
                (got / want - 1.0).abs() < 0.10,
                "{f} Hz: {got} dB/km vs {want}"
            );
        }
    }

    #[test]
    fn dry_air_absorbs_high_frequencies_more() {
        let dry = alpha_db_per_m(2000.0, 10.0, 20.0, STD_P_KPA);
        let wet = alpha_db_per_m(2000.0, 10.0, 80.0, STD_P_KPA);
        assert!(dry > wet, "{dry} vs {wet}");
    }
}
