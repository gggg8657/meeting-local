#!/usr/bin/env bash
# meeting local — 원샷 설치·실행 (macOS / Linux)
#   bash setup.sh          # venv + 의존성 + 모델 + LLM 서버 탐색 + selftest + 웹 서버 + 브라우저
#   bash setup.sh stop
# 환경변수: LLM_BASE_URL / LLM_API (서버 강제 지정), MODEL (pull 할 Ollama 모델), WHISPER_MODEL (기본 large-v3; 맥 PoC 는 small), PORT (8767)
set -euo pipefail
REPO=https://github.com/gggg8657/meeting-local.git
MODEL="${MODEL:-qwen3:8b}"; PORT="${PORT:-8767}"; export WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
if [ -t 1 ]; then B=$'\033[1m'; D=$'\033[2m'; C=$'\033[36m'; G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else B= D= C= G= R= N=; fi
STEP=0; step() { STEP=$((STEP+1)); printf '  %s[%d/8]%s %s%-14s%s ' "$D" "$STEP" "$N" "$B" "$1" "$N"; }
ok() { printf '%s✔%s %s\n' "$G" "$N" "${1:-}"; }; skip() { printf '%s–%s %s\n' "$D" "$N" "${1:-}"; }
die() { printf '%s✘ %s%s\n\n' "$R" "$*" "$N" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }; probe() { curl -fsS -m 2 "$1" >/dev/null 2>&1; }
wait_for() { for _ in $(seq 1 "${2:-30}"); do probe "$1" && return 0; sleep 1; done; return 1; }
printf '\n%s  meeting local%s  회의 녹음 → 화자 분리 전사 → 회의록·인사이트 (로컬 LLM)\n\n' "$B" "$N"

step "OS 감지"; case "$(uname -s)" in Darwin*) OS=mac ;; Linux*) OS=linux ;; *) die "지원하지 않는 OS: $(uname -s)" ;; esac; ok "$OS ($(uname -m))"
step "패키지 확보"
if [ -f "$(dirname "${BASH_SOURCE[0]:-.}")/app.py" ]; then cd "$(dirname "${BASH_SOURCE[0]}")"; ok "이미 있음 → $(pwd)"
elif [ -f ./meeting-local/app.py ]; then cd meeting-local; ok "이미 있음 → $(pwd)"
else has git || die "git 없음. 폐쇄망이면 번들을 풀고 그 폴더에서 실행"; git clone -q "$REPO" meeting-local; cd meeting-local; ok "클론 → $(pwd)"; fi
if [ "${1:-}" = "stop" ]; then [ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null && rm -f .server.pid && ok "웹 서버 종료" || skip "실행 중인 서버 없음"; exit 0; fi

step "ffmpeg"; has ffmpeg || case "$OS" in mac) has brew && brew install -q ffmpeg ;; linux) die "ffmpeg 없음: sudo apt install ffmpeg (폐쇄망은 static 빌드 반입)" ;; esac; ok "$(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)"

step "Python venv"
PY=""; for c in python3.11 python3.12 python3; do has "$c" && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)' 2>/dev/null && { PY=$c; break; }; done
[ -n "$PY" ] || die "Python 3.10~3.12 필요"
if [ ! -x venv/bin/python ]; then if has uv; then uv venv --python "$(command -v $PY)" venv -q; else "$PY" -m venv venv; fi; fi
if [ -d wheels ]; then venv/bin/pip install -q --no-index --find-links wheels -r requirements.txt   # 폐쇄망 번들
elif has uv; then VIRTUAL_ENV="$PWD/venv" uv pip install -q -r requirements.txt; else venv/bin/pip install -q -r requirements.txt; fi
if command -v nvidia-smi >/dev/null 2>&1 && [ ! -d wheels ] && ! venv/bin/python -c "import nvidia.cublas, nvidia.cudnn" 2>/dev/null; then  # GPU: ctranslate2 가 쓸 CUDA 12 cuBLAS·cuDNN
  if has uv; then VIRTUAL_ENV="$PWD/venv" uv pip install -q nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"; else venv/bin/pip install -q nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"; fi
fi
ok "$(venv/bin/python --version) + faster-whisper/sherpa-onnx"

step "모델"
mkdir -p models
if [ ! -f models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx ]; then
  curl -fsSL https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj -C models || die "화자분리 모델 다운로드 실패 (폐쇄망은 번들 사용)"; fi
[ -f models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx ] || curl -fsSL -o models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx || die "임베딩 모델 다운로드 실패"
if [ -z "${HF_HUB_OFFLINE:-}" ]; then venv/bin/python -c "from faster_whisper.utils import download_model; download_model('$WHISPER_MODEL', output_dir=None, cache_dir='models/whisper')" >/dev/null 2>&1 || die "whisper $WHISPER_MODEL 다운로드 실패"; fi
ok "whisper=$WHISPER_MODEL, 화자분리 pyannote+3D-Speaker"

step "LLM 서버"
export LLM_API="${LLM_API:-}" LLM_BASE_URL="${LLM_BASE_URL:-}"
if [ -n "$LLM_BASE_URL" ]; then [ -n "$LLM_API" ] || { case "$LLM_BASE_URL" in *11434*) LLM_API=ollama ;; *) LLM_API=openai ;; esac; }; ok "지정됨 → $LLM_API $LLM_BASE_URL"
elif probe http://localhost:11434/api/tags; then LLM_API=ollama LLM_BASE_URL=http://localhost:11434; ok "Ollama (:11434)"
else for p in 8000 1234 8080; do probe "http://localhost:$p/v1/models" && { LLM_API=openai LLM_BASE_URL="http://localhost:$p/v1"; ok "OpenAI 호환 (:$p)"; break; }; done; fi
if [ -z "$LLM_BASE_URL" ]; then
  has ollama || { case "$OS" in mac) has brew && brew install -q ollama ;; linux) curl -fsSL https://ollama.com/install.sh | sh ;; esac; }
  has ollama || die "LLM 서버가 없습니다. LLM_BASE_URL=http://gpu:8000/v1 bash setup.sh"
  nohup ollama serve >/dev/null 2>&1 & wait_for http://localhost:11434/api/tags 30 || die "Ollama 기동 실패"
  LLM_API=ollama LLM_BASE_URL=http://localhost:11434; ok "Ollama 기동"
fi
if [ "$LLM_API" = ollama ]; then
  curl -fsS "$LLM_BASE_URL/api/tags" | grep -q '"name"' || ollama pull "$MODEL"
  curl -fsS "$LLM_BASE_URL/api/tags" | grep -q "\"name\":\"$MODEL\"" || MODEL=$(curl -fsS "$LLM_BASE_URL/api/tags" | venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["models"][0]["name"])')
else MODEL=$(curl -fsS "$LLM_BASE_URL/models" | venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'); fi

step "자가검증"; venv/bin/python selftest.py >/dev/null || die "selftest 실패"; ok "병합·이름치환·분할·회의록 조립"
step "웹 서버"
[ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null || true
LLM_API=$LLM_API LLM_BASE_URL=$LLM_BASE_URL LLM_MODEL=$MODEL PORT=$PORT nohup venv/bin/python app.py > server.log 2>&1 & echo $! > .server.pid
wait_for "http://localhost:$PORT/api/models" 20 || { cat server.log; die "웹 서버 기동 실패"; }
URL="http://localhost:$PORT"; ok "$URL"
case "$OS" in mac) open "$URL" ;; linux) has xdg-open && xdg-open "$URL" >/dev/null 2>&1 || true ;; esac
printf '\n  %s준비 완료%s  %s%s%s   LLM: %s %s   종료: bash setup.sh stop   로그: server.log\n\n' "$B" "$N" "$C" "$URL" "$N" "$LLM_API" "$MODEL"
