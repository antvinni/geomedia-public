//! Every data source geomedia can use, with its licence and cost, so nothing is hidden.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SourceInfo {
    pub id: &'static str,
    pub kind: &'static str,
    pub provider: &'static str,
    pub licence: &'static str,
    pub cost: &'static str,
    pub coverage: &'static str,
    /// Used unless the user switches it off.
    pub default: bool,
    pub status: &'static str,
}

pub fn registry() -> &'static [SourceInfo] {
    const R: &[SourceInfo] = &[
        SourceInfo {
            id: "adsblol",
            kind: "ADS-B",
            provider: "adsb.lol globe_history (GitHub releases)",
            licence: "ODbL 1.0",
            cost: "free",
            coverage: "global, depends on receivers",
            default: true,
            status: "active",
        },
        SourceInfo {
            id: "tar1090-types",
            kind: "aircraft types",
            provider: "Mictronics database via tar1090-db",
            licence: "unspecified (factual ICAO type data)",
            cost: "free",
            coverage: "global",
            default: true,
            status: "active",
        },
        SourceInfo {
            id: "glo30",
            kind: "terrain",
            provider: "Copernicus DEM GLO-30 (AWS open data)",
            licence: "Copernicus DEM licence (free, attribution)",
            cost: "free",
            coverage: "global land",
            default: true,
            status: "active",
        },
        SourceInfo {
            id: "egm2008",
            kind: "geoid",
            provider: "NGA EGM2008 via PROJ-data",
            licence: "public domain",
            cost: "free",
            coverage: "global",
            default: true,
            status: "active",
        },
        SourceInfo {
            id: "nominatim",
            kind: "geocoding",
            provider: "OpenStreetMap Nominatim",
            licence: "ODbL 1.0 (1 request/s)",
            cost: "free",
            coverage: "global",
            default: true,
            status: "active",
        },
        SourceInfo {
            id: "openmeteo",
            kind: "weather",
            provider: "Open-Meteo historical forecast API (pressure levels); --weather none to switch off",
            licence: "CC BY 4.0; free API for non-commercial use",
            cost: "free",
            coverage: "global",
            default: true,
            status: "active",
        },
        SourceInfo {
            id: "opensky",
            kind: "ADS-B",
            provider: "OpenSky Network",
            licence: "research access agreement",
            cost: "free for research",
            coverage: "global, depends on receivers",
            default: false,
            status: "not yet implemented (opt-in)",
        },
    ];
    R
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_defaults() {
        let r = registry();
        for id in [
            "adsblol",
            "tar1090-types",
            "glo30",
            "egm2008",
            "nominatim",
            "openmeteo",
            "opensky",
        ] {
            assert!(r.iter().any(|s| s.id == id), "{id}");
        }
        assert!(
            r.iter()
                .all(|s| !s.licence.is_empty() && !s.coverage.is_empty())
        );
        let om = r.iter().find(|s| s.id == "openmeteo").unwrap();
        assert!(om.default && om.status == "active" && om.licence.contains("non-commercial"));
        for s in r {
            assert_eq!(
                s.default,
                s.cost == "free" && s.status == "active",
                "{}",
                s.id
            );
        }
    }
}
