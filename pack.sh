#!/usr/bin/env bash
# 폐쇄망 반입 번들 (인터넷 되는 PC에서 실행). 기본 linux-x64 / Python 3.11.
#   ./pack.sh                      # → dist-offline/meeting-local-linux-x64.tar.gz
#   WHISPER_MODEL=medium ./pack.sh  # 더 작은 ASR 모델로 (large-v3 ≈ 3GB)
set -euo pipefail; cd "$(dirname "$0")"
PLAT="${PLAT:-manylinux2014_x86_64}"; PYV="${PYV:-3.11}"; WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
STAGE=dist-offline/meeting-local-linux-x64; rm -rf "$STAGE"; mkdir -p "$STAGE/wheels" "$STAGE/models"
# 1) 휠 (대상 플랫폼용, 바이너리만). CUDA 가속은 서버에 cudnn/cublas 가 있어야 함 — 없으면 WHISPER_DEVICE=cpu
pip download -q -r requirements.txt -d "$STAGE/wheels" --platform "$PLAT" --python-version "$PYV" --only-binary=:all:
# 2) 모델: 화자분리(sherpa) + whisper CT2
cp -r models/sherpa-onnx-pyannote-segmentation-3-0 models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx "$STAGE/models/"
venv/bin/python -c "from faster_whisper.utils import download_model; download_model('$WHISPER_MODEL', cache_dir='$STAGE/models/whisper')"
# 3) 앱
cp app.py gpu_pick.py ui.html goal-prompt.md selftest.py setup.sh requirements.txt README.md NOTICE LICENSE "$STAGE/"
cat > "$STAGE/INSTALL.md" <<INS
# 폐쇄망 설치 (meeting-local)
tar -xzf meeting-local-linux-x64.tar.gz && cd meeting-local-linux-x64
# Python 3.11 + ffmpeg 가 서버에 있어야 함 (ffmpeg static: https://johnvansickle.com/ffmpeg/ 를 따로 반입)
export HF_HUB_OFFLINE=1 WHISPER_MODEL=$WHISPER_MODEL          # 모델 다운로드 시도 차단
LLM_BASE_URL=http://<gpu-host>:11434 bash setup.sh            # Ollama  (wheels/·models/ 가 있으면 다운로드 없음)
LLM_API=openai LLM_BASE_URL=http://<gpu-host>:8000/v1 bash setup.sh   # vLLM 등
# GPU 전사: WHISPER_DEVICE=cuda (CUDA 12 + cuDNN 9 필요). CPU 는 WHISPER_DEVICE=cpu (large-v3 는 느림 → medium 권장)
INS
( cd dist-offline && tar -czf meeting-local-linux-x64.tar.gz meeting-local-linux-x64 ); ls -lh dist-offline/*.tar.gz
