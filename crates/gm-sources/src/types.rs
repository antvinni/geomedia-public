//! ICAO aircraft type designator → engine class, from tar1090-db's `icao_aircraft_types2`
//! (gzip JSON `{"PC12": ["PILATUS PC-12", "L1T", "L"], …}`; `L1T` = landplane, 1 engine,
//! turboprop/turboshaft).

use std::collections::HashMap;

use serde::Serialize;

use crate::SourceError;

pub const TYPES_URL: &str =
    "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types2.js";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EngineClass {
    Piston,
    Turboprop,
    Jet,
    Electric,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TypeInfo {
    pub class: EngineClass,
    pub rotorcraft: bool,
    pub desc: String,
}

/// Parse the (gzip or plain) JSON table.
pub fn parse_type_table(data: &[u8]) -> Result<HashMap<String, TypeInfo>, SourceError> {
    let json = crate::adsblol::gunzip_bounded(data)
        .ok_or_else(|| SourceError::Invalid("type table is neither gzip nor JSON".into()))?;
    let raw: HashMap<String, Vec<serde_json::Value>> = serde_json::from_slice(&json)
        .map_err(|e| SourceError::Invalid(format!("type table: {e}")))?;
    Ok(raw
        .into_iter()
        .map(|(code, v)| {
            let desc = v.first().and_then(|x| x.as_str()).unwrap_or("").to_string();
            let d = v.get(1).and_then(|x| x.as_str()).unwrap_or("");
            let rotorcraft = d.starts_with('H');
            let class = match d.as_bytes().get(2) {
                Some(b'P') => EngineClass::Piston,
                Some(b'T') => EngineClass::Turboprop,
                Some(b'J') => EngineClass::Jet,
                Some(b'E') => EngineClass::Electric,
                _ => EngineClass::Unknown,
            };
            (
                code,
                TypeInfo {
                    class,
                    rotorcraft,
                    desc,
                },
            )
        })
        .collect())
}

/// The table, fetched once into the cache.
pub fn type_table() -> Result<HashMap<String, TypeInfo>, SourceError> {
    let dest = crate::cache_dir()
        .join("types")
        .join("icao_aircraft_types2.json.gz");
    crate::paths::touch_used("types", "icao_aircraft_types2.json.gz");
    let path = crate::fetch(
        &crate::Fetch {
            url: TYPES_URL.to_string(),
            dest,
            expected_len: None,
            timeout: None,
        },
        &mut |_, _| {},
    )?;
    let data = std::fs::read(&path).map_err(|source| SourceError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_type_table(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{"PC12":["PILATUS PC-12","L1T","L"],"B350":["BEECHCRAFT Super King Air 350","L2T","M"],
        "A320":["AIRBUS A-320","L2J","M"],"C172":["CESSNA 172 Skyhawk","L1P","L"],"R44":["ROBINSON R-44 Raven","H1P","L"],
        "EC35":["AIRBUS HELICOPTERS EC-135/635","H2T","L"],"PIPA":["PIPISTREL Alpha Electro","L1E","L"],"ODD":["?","Q"]}"#;

    #[test]
    fn parses_type_table() {
        let gz = crate::adsblol::tests::gz(FIXTURE);
        let t = parse_type_table(&gz).unwrap();
        assert_eq!(t["PC12"].class, EngineClass::Turboprop);
        assert_eq!(t["B350"].class, EngineClass::Turboprop);
        assert_eq!(t["A320"].class, EngineClass::Jet);
        assert_eq!(t["C172"].class, EngineClass::Piston);
        assert!(t["R44"].rotorcraft && t["R44"].class == EngineClass::Piston);
        assert!(t["EC35"].rotorcraft && t["EC35"].class == EngineClass::Turboprop);
        assert_eq!(t["PIPA"].class, EngineClass::Electric);
        assert_eq!(t["ODD"].class, EngineClass::Unknown);
        assert_eq!(t["A320"].desc, "AIRBUS A-320");
    }

    #[test]
    fn unknown_code() {
        let t = parse_type_table(FIXTURE.as_bytes()).unwrap();
        assert!(!t.contains_key("ZZZZ"));
        assert!(parse_type_table(b"garbage").is_err());
    }
}
