#!/bin/bash
# Fetch AeroSonicDB YPAD-0523 v1.1.1 (Zenodo record 8371595, CC BY-NC 4.0) into the kept-data
# media folder, verify every file against the record's md5, unpack the WAVs, and create the
# tools venv. Never writes into the repo except the venv (git-ignored).
#   GEOMEDIA_MEDIA overrides the media folder (default: geomedia's macOS data folder).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MEDIA="${GEOMEDIA_MEDIA:-$HOME/Library/Application Support/geomedia/media}/aerosonic"
REC=8371595
FILES=(audio.zip env_audio.zip sample_meta.csv environment_class_mappings.csv README.md LICENSE.txt)
UA="geomedia-eval"
mkdir -p "$MEDIA/download"
md5() { python3 -I -c 'import hashlib,sys; h=hashlib.md5(); f=open(sys.argv[1],"rb"); [h.update(b) for b in iter(lambda: f.read(1<<20), b"")]; print(h.hexdigest())' "$1"; }
SUMS=$(curl -fsS -A "$UA" "https://zenodo.org/api/records/$REC" \
  | python3 -I -c 'import json,sys; [print(f["key"], f["checksum"].split(":")[1]) for f in json.load(sys.stdin)["files"]]')
for f in "${FILES[@]}"; do
  want=$(awk -v k="$f" '$1==k {print $2}' <<<"$SUMS")
  [ -n "$want" ] || { echo "no checksum for $f in record $REC" >&2; exit 1; }
  dest="$MEDIA/download/$f"
  if [ ! -f "$dest" ] || [ "$(md5 "$dest")" != "$want" ]; then
    echo "fetching $f"
    curl -fL --retry 3 -A "$UA" -o "$dest" "https://zenodo.org/records/$REC/files/$f?download=1"
  fi
  got=$(md5 "$dest")
  [ "$got" = "$want" ] || { echo "md5 mismatch for $f: $got != $want" >&2; exit 1; }
  echo "ok $f"
done
cp "$MEDIA/download/"{sample_meta.csv,environment_class_mappings.csv,README.md,LICENSE.txt} "$MEDIA/"
# Unpack each archive into a fresh directory, then keep only its WAVs, flattened.
for z in audio env_audio; do
  out="$MEDIA/$z"
  if [ -d "$out" ] && [ -n "$(ls -A "$out")" ]; then
    echo "$out is not empty: not overwriting"
    continue
  fi
  tmp="$MEDIA/.unpack-$z"
  rm -rf "$tmp"
  mkdir -p "$tmp" "$out"
  unzip -q "$MEDIA/download/$z.zip" -d "$tmp"
  find "$tmp" -type f -iname '*.wav' ! -name '._*' -exec mv {} "$out/" \;
  rm -rf "$tmp"
  echo "$z: $(ls "$out" | wc -l | tr -d ' ') WAVs"
done
if [ ! -x "$HERE/.venv/bin/python" ]; then
  python3 -m venv "$HERE/.venv"
fi
"$HERE/.venv/bin/pip" install -q -r "$HERE/requirements.txt"
echo "venv: $("$HERE/.venv/bin/python" -c 'import numpy; print("numpy", numpy.__version__)')"
