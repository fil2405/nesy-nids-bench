#!/usr/bin/env bash
# Costruisce il pacchetto da dare al correlatore (sul portatile, a checkout pulito).
# Uso: bash server/build_package.sh      -> dist/nesy-<commit>/{repo.bundle,data.tar,requirements.lock,install.sh,SHA256SUMS}
# Il bundle ha un solo commit con l'ultimo stato (senza thesis/): sul server il checkout e' pulito e le run risultano citabili.
# Il commit del pacchetto non coincide con quello della repo: la corrispondenza sta in dist/nesy-<commit>/SOURCE.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain)" ] || { git status --short; echo "ERRORE: checkout sporco, fai commit" >&2; exit 1; }
COMMIT="$(git rev-parse --short HEAD)"
OUT="dist/nesy-$COMMIT"; rm -rf "$OUT"; mkdir -p "$OUT"
SNAP="$(mktemp -d)"; trap 'rm -rf "$SNAP"' EXIT
git archive HEAD | tar -x -C "$SNAP"; rm -rf "$SNAP/thesis"
GN="$(git log -1 --format=%an)"; GE="$(git log -1 --format=%ae)"
git -C "$SNAP" init -q -b main
git -C "$SNAP" add -A
git -C "$SNAP" -c user.name="$GN" -c user.email="$GE" -c commit.gpgsign=false commit -q -m "Initial commit"
git -C "$SNAP" bundle create "$PWD/$OUT/repo.bundle" HEAD >/dev/null
echo "repo $COMMIT -> pacchetto $(git -C "$SNAP" rev-parse --short HEAD)" > "$OUT/SOURCE"
cp requirements.lock "$OUT/"; cp server/install.sh "$OUT/"
[ -d data/raw/nsl-kdd ] || { echo "ERRORE: manca data/raw/nsl-kdd" >&2; exit 1; }
ls data/interim/*.parquet >/dev/null 2>&1 || echo "ATTENZIONE: data/interim/ senza Parquet: il pacchetto non ha i CIC" >&2
tar -cf "$OUT/data.tar" --exclude=.gitkeep -C data raw/nsl-kdd interim
(cd "$OUT" && sha256sum * > SHA256SUMS)
echo "commit $COMMIT"; du -sh "$OUT"/*; echo "pacchetto in $OUT"
