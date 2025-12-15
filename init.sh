#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 파라미터
# -----------------------------
PROJECT_NAME="${PROJECT_NAME:-fashion_recommand}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
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
      ca-certificates build-essential cmake pkgconf \
      python3-venv curl gnupg locales

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
# 요구사항 파일 준비
# -----------------------------
pip install -U pip setuptools wheel
info "Python 패키지 설치: $REQ_FILE"
pip install -r "$REQ_FILE"
info "pre-commit 훅 설치"
pip install pre-commit
pre-commit --version
pre-commit install
info "[$PROJECT_NAME] init 완료"
