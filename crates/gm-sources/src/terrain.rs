//! Ground heights: Copernicus GLO-30 (EGM2008 orthometric) + the EGM2008 geoid undulation N,
//! giving WGS84 ellipsoidal heights. Ocean has no GLO-30 tile: height there is N.

use std::collections::HashMap;
use std::path::Path;

use crate::SourceError;
use crate::area::Area;

pub const GEOID_URL: &str = "https://cdn.proj.org/us_nga_egm08_25.tif";
pub const GEOID_FILE: &str = "us_nga_egm08_25.tif";

/// GLO-30 tile name and URL for the 1°×1° cell whose south-west corner is (lat, lon).
pub fn glo30_tile(lat_floor: i32, lon_floor: i32) -> (String, String) {
    let ns = if lat_floor < 0 { 'S' } else { 'N' };
    let ew = if lon_floor < 0 { 'W' } else { 'E' };
    let name = format!(
        "Copernicus_DSM_COG_10_{ns}{:02}_00_{ew}{:03}_00_DEM",
        lat_floor.unsigned_abs(),
        lon_floor.unsigned_abs()
    );
    let url = format!("https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif");
    (name, url)
}

/// A tile that doesn't exist (GLO-30 has none over open water): HTTP 403/404 only.
pub fn is_missing_tile(e: &SourceError) -> bool {
    matches!(
        e,
        SourceError::HttpStatus {
            status: 403 | 404,
            ..
        }
    )
}

/// A georeferenced single-band f32 raster (GeoTIFF).
pub struct Raster {
    width: usize,
    height: usize,
    data: Vec<f32>,
    /// Longitude/latitude of pixel (0,0)'s centre, and pixel size in degrees.
    lon0: f64,
    lat0: f64,
    dlon: f64,
    dlat: f64,
    nodata: Option<f32>,
}

impl Raster {
    pub fn open(path: &Path) -> Result<Raster, SourceError> {
        use tiff::decoder::{Decoder, DecodingResult};
        use tiff::tags::Tag;
        let bad = |why: String| SourceError::Invalid(format!("{}: {why}", path.display()));
        let file = std::fs::File::open(path).map_err(|source| SourceError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut d = Decoder::new(std::io::BufReader::new(file)).map_err(|e| bad(e.to_string()))?;
        let (w, h) = d.dimensions().map_err(|e| bad(e.to_string()))?;
        let scale = d
            .get_tag_f64_vec(Tag::ModelPixelScaleTag)
            .map_err(|e| bad(format!("no pixel scale: {e}")))?;
        let tie = d
            .get_tag_f64_vec(Tag::ModelTiepointTag)
            .map_err(|e| bad(format!("no tie point: {e}")))?;
        if scale.len() < 2 || tie.len() < 6 || scale[0] <= 0.0 || scale[1] <= 0.0 {
            return Err(bad("unsupported georeferencing".into()));
        }
        // GTRasterTypeGeoKey (1025): 1 = PixelIsArea (default), 2 = PixelIsPoint.
        let is_point = d
            .get_tag_u16_vec(Tag::GeoKeyDirectoryTag)
            .ok()
            .and_then(|k| {
                k.get(4..).map(|keys| {
                    keys.as_chunks::<4>()
                        .0
                        .iter()
                        .any(|e| e[0] == 1025 && e[1] == 0 && e[3] == 2)
                })
            })
            .unwrap_or(false);
        let nodata = d
            .find_tag(Tag::Unknown(42113))
            .ok()
            .flatten()
            .and_then(|v| v.into_string().ok())
            .and_then(|s| s.trim_end_matches('\0').trim().parse::<f32>().ok());
        let data: Vec<f32> = match d.read_image().map_err(|e| bad(e.to_string()))? {
            DecodingResult::F32(v) => v,
            DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
            DecodingResult::I16(v) => v.into_iter().map(f32::from).collect(),
            DecodingResult::U16(v) => v.into_iter().map(f32::from).collect(),
            DecodingResult::I32(v) => v.into_iter().map(|x| x as f32).collect(),
            _ => return Err(bad("unsupported sample type".into())),
        };
        let (width, height) = (w as usize, h as usize);
        if data.len() < width * height {
            return Err(bad("truncated raster".into()));
        }
        let (dlon, dlat) = (scale[0], scale[1]);
        let (west, north) = (tie[3] - tie[0] * dlon, tie[4] + tie[1] * dlat);
        let half = if is_point { 0.0 } else { 0.5 };
        Ok(Raster {
            width,
            height,
            data,
            lon0: west + half * dlon,
            lat0: north - half * dlat,
            dlon,
            dlat,
            nodata,
        })
    }

    /// Like `sample`, but within one pixel beyond the edge clamps to the edge (tile seams).
    pub fn sample_edge(&self, lat: f64, lon: f64) -> Option<f64> {
        self.sample(lat, lon).or_else(|| {
            let clamp_lat = (self.lat0 - lat) / self.dlat;
            let py = clamp_lat.clamp(0.0, self.height as f64 - 1.0);
            let lon_px = [lon, lon - 360.0, lon + 360.0]
                .into_iter()
                .map(|l| (l - self.lon0) / self.dlon)
                .find(|p| *p >= -1.0 && *p <= self.width as f64)?;
            if !(-1.0..=self.height as f64).contains(&clamp_lat) {
                return None;
            }
            let px = lon_px.clamp(0.0, self.width as f64 - 1.0);
            self.sample(self.lat0 - py * self.dlat, self.lon0 + px * self.dlon)
        })
    }

    /// Bilinear sample; None outside the raster or on no-data.
    pub fn sample(&self, lat: f64, lon: f64) -> Option<f64> {
        let (w, h) = (self.width as f64, self.height as f64);
        // Try the longitude as given, then wrapped by ±360 (global grids).
        let px = [lon, lon - 360.0, lon + 360.0]
            .into_iter()
            .map(|l| (l - self.lon0) / self.dlon)
            .find(|p| *p >= -0.5 && *p <= w - 0.5)?;
        let py = (self.lat0 - lat) / self.dlat;
        if !(py >= -0.5 && py <= h - 0.5) {
            return None;
        }
        let (px, py) = (px.clamp(0.0, w - 1.0), py.clamp(0.0, h - 1.0));
        let (x0, y0) = (px.floor() as usize, py.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (px - x0 as f64, py - y0 as f64);
        let v = |x: usize, y: usize| -> Option<f64> {
            let z = *self.data.get(y * self.width + x)?;
            (z.is_finite() && self.nodata != Some(z)).then_some(f64::from(z))
        };
        let (a, b, c, d) = (v(x0, y0)?, v(x1, y0)?, v(x0, y1)?, v(x1, y1)?);
        Some(a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy)
    }
}

/// Terrain for an area: the GLO-30 tiles it touches plus the geoid.
pub struct Terrain {
    tiles: HashMap<(i32, i32), Option<Raster>>,
    geoid: Option<Raster>,
}

/// What `Terrain::ensure` would need to download.
pub struct TerrainPlan {
    pub tiles: Vec<(i32, i32)>,
    pub missing_tiles: Vec<(i32, i32)>,
    /// Tiles with no GLO-30 data (open water, or withheld): treated as sea level.
    pub ocean_tiles: Vec<(i32, i32)>,
    pub geoid_cached: bool,
}

impl Terrain {
    pub fn plan(area: &Area) -> TerrainPlan {
        let (s, w, n, e) = area.bbox();
        let dir = crate::cache_dir().join("dem/glo30");
        let mut tiles = Vec::new();
        // Integer longitude cells, normalised to [-180, 179] (circle boxes may run past ±180).
        let norm = |k: i32| (k + 180).rem_euclid(360) - 180;
        let mut lons: Vec<i32> = if w <= e {
            (w.floor() as i32..=e.floor() as i32).map(norm).collect()
        } else {
            (w.floor() as i32..=179)
                .chain(-180..=e.floor() as i32)
                .collect()
        };
        lons.sort_unstable();
        lons.dedup();
        for la in s.floor() as i32..=n.floor() as i32 {
            for &lo in &lons {
                tiles.push((la, lo));
            }
        }
        let missing_tiles = tiles
            .iter()
            .copied()
            .filter(|&(la, lo)| {
                let name = glo30_tile(la, lo).0;
                !dir.join(format!("{name}.tif")).exists()
                    && !dir.join(format!("{name}.ocean")).exists()
            })
            .collect();
        let ocean_tiles = tiles
            .iter()
            .copied()
            .filter(|&(la, lo)| dir.join(format!("{}.ocean", glo30_tile(la, lo).0)).exists())
            .collect();
        TerrainPlan {
            tiles,
            missing_tiles,
            ocean_tiles,
            geoid_cached: crate::cache_dir().join("geoid").join(GEOID_FILE).exists(),
        }
    }

    /// Download (with `consent`) what `area` needs, without loading anything into memory.
    pub fn download(area: &Area, consent: bool) -> Result<TerrainPlan, SourceError> {
        let plan = Terrain::plan(area);
        let need = plan.missing_tiles.len() as u64 * (30 << 20)
            + if plan.geoid_cached { 0 } else { 81 << 20 };
        if need > 0 && !consent {
            return Err(SourceError::NeedsConsent {
                what: "terrain + geoid".into(),
                size: format!("~{}", crate::human_size(need)),
            });
        }
        let dir = crate::cache_dir().join("dem/glo30");
        for &(la, lo) in &plan.missing_tiles {
            let (name, url) = glo30_tile(la, lo);
            match crate::fetch(
                &crate::Fetch {
                    url,
                    dest: dir.join(format!("{name}.tif")),
                    expected_len: None,
                    timeout: None,
                },
                &mut |_, _| {},
            ) {
                Ok(_) => {}
                // GLO-30 has no tiles over open water (and withholds a few land tiles); remember it.
                Err(e) if is_missing_tile(&e) => {
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = std::fs::write(dir.join(format!("{name}.ocean")), b"");
                }
                Err(e) => return Err(e),
            }
        }
        if !plan.geoid_cached {
            let gpath = crate::cache_dir().join("geoid").join(GEOID_FILE);
            crate::fetch(
                &crate::Fetch {
                    url: GEOID_URL.into(),
                    dest: gpath,
                    expected_len: None,
                    timeout: None,
                },
                &mut |_, _| {},
            )?;
        }
        Ok(Terrain::plan(area))
    }

    /// Download (with `consent`) and load the tiles and geoid for `area`.
    pub fn ensure(area: &Area, consent: bool) -> Result<Terrain, SourceError> {
        let plan = Terrain::download(area, consent)?;
        let dir = crate::cache_dir().join("dem/glo30");
        let mut tiles = HashMap::new();
        for (la, lo) in plan.tiles {
            let name = glo30_tile(la, lo).0;
            let tif = dir.join(format!("{name}.tif"));
            tiles.insert(
                (la, lo),
                if tif.exists() {
                    crate::paths::touch_used("terrain", &name);
                    Some(Raster::open(&tif)?)
                } else {
                    None
                },
            );
        }
        let gpath = crate::cache_dir().join("geoid").join(GEOID_FILE);
        crate::paths::touch_used("geoid", GEOID_FILE);
        Ok(Terrain {
            tiles,
            geoid: Some(Raster::open(&gpath)?),
        })
    }

    /// Build from already-loaded parts (tests, offline use).
    pub fn from_parts(
        tiles: HashMap<(i32, i32), Option<Raster>>,
        geoid: Option<Raster>,
    ) -> Terrain {
        Terrain { tiles, geoid }
    }

    /// Geoid undulation N (ellipsoidal − orthometric height); 0 if the geoid isn't loaded.
    pub fn geoid_n(&self, lat: f64, lon: f64) -> f64 {
        self.geoid
            .as_ref()
            .and_then(|g| g.sample(lat, lon))
            .unwrap_or(0.0)
    }

    /// Ellipsoidal ground height: DEM + N; ocean (no tile) = N; unknown geoid = DEM only.
    pub fn ground_h_ell(&self, lat: f64, lon: f64) -> f64 {
        let key = (lat.floor() as i32, lon.floor() as i32);
        let dem = self
            .tiles
            .get(&key)
            .and_then(|t| t.as_ref())
            .and_then(|r| r.sample(lat, lon))
            .unwrap_or(0.0);
        let n = self
            .geoid
            .as_ref()
            .and_then(|g| g.sample(lat, lon))
            .unwrap_or(0.0);
        dem + n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiff::encoder::{TiffEncoder, colortype::Gray32Float};
    use tiff::tags::Tag;

    /// A w×h f32 GeoTIFF with the given top-left corner and pixel size (pixel-is-area).
    #[allow(clippy::too_many_arguments)]
    fn write_geotiff(
        path: &Path,
        w: u32,
        h: u32,
        data: &[f32],
        west: f64,
        north: f64,
        dx: f64,
        dy: f64,
        point: bool,
    ) {
        let mut enc = TiffEncoder::new(std::fs::File::create(path).unwrap()).unwrap();
        let mut img = enc.new_image::<Gray32Float>(w, h).unwrap();
        img.encoder()
            .write_tag(Tag::ModelPixelScaleTag, &[dx, dy, 0.0][..])
            .unwrap();
        img.encoder()
            .write_tag(
                Tag::ModelTiepointTag,
                &[0.0, 0.0, 0.0, west, north, 0.0][..],
            )
            .unwrap();
        // GeoKeyDirectory: version 1.1.0, 1 key: GTRasterTypeGeoKey (1025) = 1 area / 2 point.
        let raster_type = if point { 2u16 } else { 1u16 };
        img.encoder()
            .write_tag(
                Tag::GeoKeyDirectoryTag,
                &[1u16, 1, 0, 1, 1025, 0, 1, raster_type][..],
            )
            .unwrap();
        img.write_data(data).unwrap();
    }

    #[test]
    fn raster_bilinear() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.tif");
        // 3×3 grid, pixel 1°, top-left corner (10E, 52N): centres at 10.5/11.5/12.5 E, 51.5/50.5/49.5 N.
        write_geotiff(
            &p,
            3,
            3,
            &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            10.0,
            52.0,
            1.0,
            1.0,
            false,
        );
        let r = Raster::open(&p).unwrap();
        assert!((r.sample(51.5, 10.5).unwrap() - 0.0).abs() < 1e-9);
        assert!((r.sample(50.5, 11.5).unwrap() - 40.0).abs() < 1e-9);
        assert!(
            (r.sample(51.0, 11.0).unwrap() - 20.0).abs() < 1e-9,
            "midpoint of 0,10,30,40"
        );
        let q = dir.path().join("b.tif");
        write_geotiff(
            &q,
            3,
            3,
            &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            10.0,
            52.0,
            1.0,
            1.0,
            true,
        );
        let r = Raster::open(&q).unwrap();
        assert!(
            (r.sample(52.0, 10.0).unwrap() - 0.0).abs() < 1e-9,
            "pixel-is-point: tie point is a pixel centre"
        );
    }

    #[test]
    fn raster_outside_none() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.tif");
        write_geotiff(&p, 2, 2, &[1.0, 2.0, 3.0, 4.0], 10.0, 52.0, 1.0, 1.0, false);
        let r = Raster::open(&p).unwrap();
        assert!(r.sample(40.0, 10.5).is_none() && r.sample(51.5, 30.0).is_none());
        assert!(Raster::open(&dir.path().join("missing.tif")).is_err());
    }

    #[test]
    fn tile_names() {
        let (n, u) = glo30_tile(40, -74);
        assert_eq!(n, "Copernicus_DSM_COG_10_N40_00_W074_00_DEM");
        assert_eq!(
            u,
            "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N40_00_W074_00_DEM/Copernicus_DSM_COG_10_N40_00_W074_00_DEM.tif"
        );
        assert_eq!(
            glo30_tile(-34, 151).0,
            "Copernicus_DSM_COG_10_S34_00_E151_00_DEM"
        );
        assert_eq!(
            glo30_tile(0, 0).0,
            "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"
        );
    }

    #[test]
    fn ocean_is_geoid() {
        let dir = tempfile::tempdir().unwrap();
        let g = dir.path().join("g.tif");
        write_geotiff(
            &g,
            2,
            2,
            &[-30.0, -30.0, -30.0, -30.0],
            -75.0,
            42.0,
            1.0,
            1.0,
            false,
        );
        let mut tiles = HashMap::new();
        tiles.insert((40, -74), None); // known ocean / no tile
        let t = Terrain::from_parts(tiles, Some(Raster::open(&g).unwrap()));
        assert!((t.ground_h_ell(40.5, -73.5) + 30.0).abs() < 1e-9);
    }

    #[test]
    fn ocean_only_on_missing_status() {
        let missing = SourceError::HttpStatus {
            url: "u".into(),
            status: 404,
        };
        let interrupted = SourceError::Http {
            url: "u".into(),
            reason: "interrupted after 24031234 bytes (reset); run again to resume".into(),
        };
        let forbidden = SourceError::HttpStatus {
            url: "u".into(),
            status: 403,
        };
        let server = SourceError::HttpStatus {
            url: "u".into(),
            status: 503,
        };
        assert!(is_missing_tile(&missing) && is_missing_tile(&forbidden));
        assert!(!is_missing_tile(&interrupted) && !is_missing_tile(&server));
    }

    #[test]
    fn antimeridian_tile_plan() {
        let a = Area::Circle {
            lat: -16.8,
            lon: 179.9,
            radius_m: 30_000.0,
        };
        let tiles = Terrain::plan(&a).tiles;
        assert!(
            tiles.contains(&(-17, 179)) && tiles.contains(&(-17, -180)),
            "{tiles:?}"
        );
        assert!(
            tiles.iter().all(|&(_, lo)| (-180..=179).contains(&lo)),
            "{tiles:?}"
        );
    }

    #[test]
    fn point_grid_edges_clamp() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("p.tif");
        // Pixel-is-point 3×3 at 0.5° spacing: centres 10.0..11.0 E, 52.0..51.0 N.
        write_geotiff(
            &p,
            3,
            3,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            10.0,
            52.0,
            0.5,
            0.5,
            true,
        );
        let r = Raster::open(&p).unwrap();
        // Just past the last centre (the next tile's seam): clamp to the edge, don't report no data.
        assert!(
            r.sample(50.70, 11.0).is_none(),
            "plain sample: beyond the half-pixel margin"
        );
        assert!((r.sample_edge(50.70, 11.0).unwrap() - 9.0).abs() < 1e-6);
        assert!(
            r.sample_edge(49.0, 11.0).is_none(),
            "far outside is still none"
        );
    }

    #[test]
    fn download_does_not_load_rasters() {
        // A cache whose tile and geoid files exist but are not valid GeoTIFFs: downloading (a
        // no-op here) must not parse them; loading does.
        let dir = tempfile::tempdir().unwrap();
        let _g = crate::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("GEOMEDIA_CACHE");
        unsafe { std::env::set_var("GEOMEDIA_CACHE", dir.path()) };
        std::fs::create_dir_all(dir.path().join("dem/glo30")).unwrap();
        std::fs::create_dir_all(dir.path().join("geoid")).unwrap();
        std::fs::write(
            dir.path()
                .join("dem/glo30")
                .join(format!("{}.tif", glo30_tile(40, -74).0)),
            b"x",
        )
        .unwrap();
        std::fs::write(dir.path().join("geoid").join(GEOID_FILE), b"x").unwrap();
        let a = Area::Circle {
            lat: 40.5,
            lon: -73.5,
            radius_m: 1000.0,
        };
        let dl = Terrain::download(&a, true);
        let load = Terrain::ensure(&a, true);
        match prev {
            Some(v) => unsafe { std::env::set_var("GEOMEDIA_CACHE", v) },
            None => unsafe { std::env::remove_var("GEOMEDIA_CACHE") },
        }
        assert!(dl.is_ok(), "{:?}", dl.err());
        assert!(load.is_err());
    }

    #[test]
    fn cached_glo30_reads() {
        // Local smoke check against real cached data; skipped where the cache lacks it.
        let dem = crate::cache_dir().join("dem/glo30/Copernicus_DSM_COG_10_N40_00_W074_00_DEM.tif");
        let geoid = crate::cache_dir().join("geoid").join(GEOID_FILE);
        if !(dem.exists() && geoid.exists()) {
            eprintln!("skipped: no cached Long Island tile/geoid");
            return;
        }
        let d = Raster::open(&dem).unwrap();
        let n = Raster::open(&geoid).unwrap();
        // Mid Long Island: tens of metres orthometric; EGM2008 N there is about −31 m.
        let h = d.sample(40.80, -73.50).unwrap();
        let un = n.sample(40.80, -73.50).unwrap();
        assert!((0.0..120.0).contains(&h), "dem {h}");
        assert!((-36.0..-26.0).contains(&un), "geoid {un}");
    }
}
