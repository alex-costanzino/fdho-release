#!/usr/bin/env bash
#
# Download the meshes used for the SDF fitting experiments from the
# Stanford 3D Scanning Repository (https://graphics.stanford.edu/data/3Dscanrep/).
#
# The meshes are redistributed by Stanford under their own terms; please cite the
# repository if you use them. This script only fetches them, it does not modify them.
#
# Usage: bash scripts/download_sdf_data.sh [shape ...]
#        with no arguments, all four shapes are downloaded.

set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/sdf"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$DEST"

declare -A URLS=(
  [armadillo]="https://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
  [dragon]="https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz"
  [lucy]="https://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz"
  [thai_statue]="https://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz"
)

SHAPES=("$@")
if [ ${#SHAPES[@]} -eq 0 ]; then
  SHAPES=(armadillo dragon lucy thai_statue)
fi

for shape in "${SHAPES[@]}"; do
  url="${URLS[$shape]:-}"
  if [ -z "$url" ]; then
    echo "Unknown shape '$shape'. Available: ${!URLS[*]}" >&2
    exit 1
  fi

  if [ -f "$DEST/$shape.ply" ]; then
    echo "[$shape] already present at $DEST/$shape.ply, skipping."
    continue
  fi

  echo "[$shape] downloading $url"
  archive="$TMP/$(basename "$url")"
  curl -L --fail --progress-bar -o "$archive" "$url"

  echo "[$shape] extracting"
  work="$TMP/$shape"
  mkdir -p "$work"
  case "$archive" in
    *.tar.gz) tar -xzf "$archive" -C "$work" ;;
    *.ply.gz) gunzip -c "$archive" > "$work/$shape.ply" ;;
    *) echo "Unhandled archive type: $archive" >&2; exit 1 ;;
  esac

  # The archives differ in layout, so locate the largest .ply and use that.
  ply="$(find "$work" -name '*.ply' -type f -exec ls -S {} + | head -1)"
  if [ -z "$ply" ]; then
    echo "[$shape] no .ply found in the archive" >&2
    exit 1
  fi
  mv "$ply" "$DEST/$shape.ply"
  echo "[$shape] -> $DEST/$shape.ply"
done

echo
echo "Done. Now run:  python scripts/prepare_sdf_data.py"
