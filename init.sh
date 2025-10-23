#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 파라미터
# -----------------------------
PROJECT_NAME="${PROJECT_NAME:-fashion_recommand}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"     # 없으면 자동 생성
USE_DOCKER="${USE_DOCKER:-0}"                # 1이면 compose까지 올림
USE_NODE="${USE_NODE:-0}"                    # 1이면 node 의존 설치
SETUP_SYSTEM_DEPS="${SETUP_SYSTEM_DEPS:-1}"  # 1이면 apt 설치 진행(WSL/Ubuntu)

# -----------------------------
# 유틸
# -----------------------------
info(){ echo "▶ $*"; }
warn(){ echo "⚠️  $*"; }
die(){ echo "❌ $*"; exit 1; }

info "[$PROJECT_NAME] init 시작 (step 1)"

# -----------------------------
# 환경 감지
# -----------------------------
OS="$(uname -s || true)"
ARCH="$(uname -m || true)"
IS_WSL=0
grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1

info "프로젝트: $PROJECT_NAME, OS=$OS, ARCH=$ARCH, WSL=$IS_WSL"

# -----------------------------
# 시스템 의존성 (옵션)
# -----------------------------
if command -v apt >/dev/null 2>&1; then
  HAS_APT=1
else
  HAS_APT=0
fi

if [[ "$SETUP_SYSTEM_DEPS" == "1" ]]; then
  if [[ "$HAS_APT" == "1" ]]; then
    info "APT 패키지 설치(필요 시)"
    # 비대화형 보장(로케일/키보드 질문 방지)
    export DEBIAN_FRONTEND=noninteractive

    sudo apt update -y
    sudo apt install -y \
      curl gnupg lsb-release ca-certificates \
      gcc g++ make cmake build-essential clang llvm pkgconf texinfo \
      gcc-multilib g++-multilib \
      libbpf-dev libelf-dev linux-tools-common linux-tools-generic \
      gawk bison gettext \
      sqlite3 libsqlite3-dev \
      zlib1g-dev libzstd-dev \
      docker.io \
      openjdk-17-jdk locales python3-venv

    # 로케일(필요시)
    sudo locale-gen en_US.UTF-8 || true

    # 도커를 sudo 없이 쓰고 싶다면(선택):
    # sudo usermod -aG docker "$USER" || true
    # newgrp docker || true
  else
    warn "apt 미발견: 시스템 패키지 설치 스킵"
  fi
else
  info "SETUP_SYSTEM_DEPS=0 → 시스템 패키지 설치 스킵"
fi

# -----------------------------
# Python 가상환경
# -----------------------------
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python3.11 미설치 또는 PATH 문제"

if [[ ! -d "$VENV_DIR" ]]; then
  info "venv 생성: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"



# -----------------------------
# 요구사항 파일 준비(없으면 생성)
# -----------------------------
if [[ ! -f "$REQ_FILE" ]]; then
  warn "$REQ_FILE 이 없어 기본 템플릿을 생성합니다."
  cat > "$REQ_FILE" <<'EOF'
# 기본 예시 — 필요에 맞게 수정하세요
# PyTorch CUDA 휠 인덱스
--extra-index-url https://download.pytorch.org/whl/cu121

# Core DL (CUDA)
torch==2.4.0          #######버전확인 일단 구버전 설치
torchvision==0.19.0   #######버전확인 일단 구버전 설치

# Utils / Numerics
numpy==1.23.5
Pillow==10.4.0

# Embedding / Models
open_clip_torch==2.26.1   #######버전확인 일단 구버전 설치
timm==1.0.9
transformers==4.45.2
tokenizers==0.20.0
safetensors==0.4.5

# Vector Search (GPU)
faiss-gpu-cu12==1.12.0
EOF
fi

info "Python 패키지 설치: $REQ_FILE"
pip install -r "$REQ_FILE"
