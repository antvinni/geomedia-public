mod common;

#[test]
fn sources_lists_all() {
    let dir = tempfile::tempdir().unwrap();
    let out = common::geomedia()
        .arg("sources")
        .env("GEOMEDIA_CACHE", dir.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    for id in [
        "adsblol",
        "tar1090-types",
        "glo30",
        "egm2008",
        "nominatim",
        "openmeteo",
        "opensky",
    ] {
        assert!(s.contains(id), "{id} missing:\n{s}");
    }
    assert!(s.contains("ODbL") && s.contains("non-commercial"), "{s}");
    assert!(s.contains("cache: "), "{s}");
}
