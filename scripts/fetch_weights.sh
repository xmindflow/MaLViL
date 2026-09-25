#!/usr/bin/env bash
# Download released MaLViL checkpoints and check them against SHA256.
#
#   ./scripts/fetch_weights.sh                 # all six datasets
#   ./scripts/fetch_weights.sh synapse         # one dataset
#
# Override the source with WEIGHTS_BASE_URL=... (for example a local mirror).
# Files land in weights/, which is gitignored.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${WEIGHTS_TAG:-checkpoints}"
BASE="${WEIGHTS_BASE_URL:-https://github.com/xmindflow/MaLViL/releases/download/$TAG}"
DEST="$REPO/weights"

declare -A SUMS=(
  [malvil-ph2.pth]=70744be92b866478a9680d7cbbd659bdbd4eea542dd49ee341ffb614b9e0b4f5
  [malvil-busi.pth]=dc9bd8d215d5199d2a2cd2c965839ef877f91a57ba572cf1f1c2908952a0f73e
  [malvil-isic2017.pth]=06ff21487751f1b85fe367a1d129bb08aa4e9891d491835a6998f7f664f1de66
  [malvil-isic2018.pth]=505c6c9ef50ac4138cd360f333711090d6c77e4be9e2c488057f4d9a092804e0
  [malvil-ham10000.pth]=86ab9d3f1b53b07ea37a8f7ed9a01f2478e4628589054dcdfb08b66db640922f
  [malvil-synapse.pth]=5c0eb9cb97c90c1939c700f1ef5cd66708b309909071f145234a8dbb32b0ab56
)

case "${1:-all}" in
  all) assets=(malvil-ph2.pth malvil-busi.pth malvil-isic2017.pth malvil-isic2018.pth malvil-ham10000.pth malvil-synapse.pth) ;;
  ph2|busi|isic2017|isic2018|ham10000|synapse) assets=("malvil-$1.pth") ;;
  -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
  *) echo "Unknown target: $1 (use: all | ph2 | busi | isic2017 | isic2018 | ham10000 | synapse)" >&2; exit 1 ;;
esac

mkdir -p "$DEST"
for asset in "${assets[@]}"; do
  dest="$DEST/$asset"
  if [[ -f "$dest" ]] && echo "${SUMS[$asset]}  $dest" | sha256sum -c - >/dev/null 2>&1; then
    echo ">>> $asset already present"
    continue
  fi
  echo ">>> downloading $asset"
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$BASE/$asset"
  mv "$dest.part" "$dest"
  echo ">>> verifying $asset"
  echo "${SUMS[$asset]}  $dest" | sha256sum -c -
done

echo
echo "Checkpoints in $DEST. Score one with:"
echo "  EVAL_PT=$DEST/malvil-isic2018.pth bash scripts/main.sh ISIC2018 TEST"
