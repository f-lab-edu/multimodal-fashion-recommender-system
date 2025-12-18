# scripts/fetch_assets.sh
set -eu
: "${OWNER:=f-lab-edu}"
: "${REPO:=multimodal-fashion-recommender-system}"
: "${DATA_ASSETS_VER:=v0.1.0}"
: "${MODEL_ASSETS_VER:=m0.1.0}"

DATA_ASSET="assets-data-${DATA_ASSETS_VER}.tar.gz"
MODEL_ASSET="assets-model-${MODEL_ASSETS_VER}.tar.gz"

BASE="https://github.com/${OWNER}/${REPO}/releases/download"

mkdir -p /work/data /work/models

if [ -f "/work/data/.ready-${DATA_ASSETS_VER}" ]; then
  echo "[fetch] data ready: ${DATA_ASSETS_VER}"
else
  echo "[fetch] download ${DATA_ASSET}"
  curl -fL -o "/tmp/${DATA_ASSET}" "${BASE}/${DATA_ASSETS_VER}/${DATA_ASSET}"
  rm -rf /work/data/*
  tar -xzf "/tmp/${DATA_ASSET}" -C /work/data
  touch "/work/data/.ready-${DATA_ASSETS_VER}"
fi

if [ -f "/work/models/.ready-${MODEL_ASSETS_VER}" ]; then
  echo "[fetch] models ready: ${MODEL_ASSETS_VER}"
else
  echo "[fetch] download ${MODEL_ASSET}"
  curl -fL -o "/tmp/${MODEL_ASSET}" "${BASE}/${MODEL_ASSETS_VER}/${MODEL_ASSET}"
  rm -rf /work/models/*
  tar -xzf "/tmp/${MODEL_ASSET}" -C /work/models
  touch "/work/models/.ready-${MODEL_ASSETS_VER}"
fi

echo "[fetch] done"
