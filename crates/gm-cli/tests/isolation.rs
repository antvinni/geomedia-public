//! No CLI test may start the binary without the isolated environment of `common::geomedia()`:
//! a bare run would read and migrate the user's real data folders.

#[test]
fn tests_never_start_the_binary_directly() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut offenders = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "rs")
            && p.file_name().is_some_and(|n| n != "isolation.rs")
        {
            let text = std::fs::read_to_string(&p).unwrap();
            if text.contains("cargo_bin(\"geomedia\")") {
                offenders.push(p.display().to_string());
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "use common::geomedia(): {offenders:?}"
    );
}
