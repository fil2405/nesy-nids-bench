#!/usr/bin/env bash
# Sul server, dentro la cartella del pacchetto: bash install.sh [cartella di destinazione]
# Clona il bundle (checkout pulito), mette i dati in data/interim, crea il venv dal lock file.
# TODO: se il server ha un driver/CUDA diverso da quello che TensorFlow 2.21 vuole, qui serve
#       un'immagine Docker o un altro lock file (dipende dalla risposta del correlatore).
set -euo pipefail
PKG="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/nesy-nids-bench}"
(cd "$PKG" && sha256sum -c SHA256SUMS)
[ -e "$DEST" ] && { echo "ERRORE: $DEST esiste gia'" >&2; exit 1; }
git clone "$PKG/repo.bundle" "$DEST"
cd "$DEST"
[ -f "$PKG/data_interim.tar" ] && tar -xf "$PKG/data_interim.tar" -C data
python3 -m venv .venv
.venv/bin/pip install -r "$PKG/requirements.lock"
.venv/bin/pip install -e . --no-deps
echo "ok. Poi: source .venv/bin/activate && bash server/00_smoke.sh"
