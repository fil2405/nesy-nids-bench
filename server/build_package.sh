#!/usr/bin/env bash
# Costruisce il pacchetto da dare al correlatore (sul portatile, a checkout pulito).
# Uso: bash server/build_package.sh      -> dist/nesy-<commit>/{repo.bundle,data.tar,requirements.lock,install.sh,SHA256SUMS}
# Il bundle porta la storia git: sul server il checkout e' pulito e le run risultano citabili.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain)" ] || { git status --short; echo "ERRORE: checkout sporco, fai commit" >&2; exit 1; }
COMMIT="$(git rev-parse --short HEAD)"
OUT="dist/nesy-$COMMIT"; rm -rf "$OUT"; mkdir -p "$OUT"
git bundle create "$OUT/repo.bundle" HEAD >/dev/null
cp requirements.lock "$OUT/"; cp server/install.sh "$OUT/"
[ -d data/raw/nsl-kdd ] || { echo "ERRORE: manca data/raw/nsl-kdd" >&2; exit 1; }
ls data/interim/*.parquet >/dev/null 2>&1 || echo "ATTENZIONE: data/interim/ senza Parquet: il pacchetto non ha i CIC" >&2
tar -cf "$OUT/data.tar" --exclude=.gitkeep -C data raw/nsl-kdd interim
(cd "$OUT" && sha256sum * > SHA256SUMS)
echo "commit $COMMIT"; du -sh "$OUT"/*; echo "pacchetto in $OUT"
