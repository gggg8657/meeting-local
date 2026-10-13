# 제3자 구성요소 고지 (THIRD_PARTY_NOTICES)

meeting-local 의 원저작물은 DongJu Kim(gggg8657)이 작성했습니다(`NOTICE`, `LICENSE`). 아래 제3자 구성요소는 각자의 라이선스를 따르며, 이 저장소의 라이선스(PolyForm Noncommercial 1.0.0)로 바뀌지 않습니다.
조사일: 2026-10-12 (실제 파일·`venv` 설치 메타데이터·setup.sh·pack.sh 기준). "상업 이용" 칸은 확인한 라이선스 문구를 요약한 것이며 법률 판단이 아닙니다.

이 저장소에는 제3자 코드·가중치가 **들어 있지 않습니다**(`models/`, `venv/` 는 `.gitignore`). 아래는 `setup.sh` 가 설치하거나 내려받는 것입니다. `pack.sh` 로 만든 폐쇄망 번들은 휠과 가중치를 함께 묶으므로(재배포) 번들을 외부에 넘길 때는 아래 라이선스를 같이 전달하세요.

## 직접 의존성 (`requirements.txt`)

| 이름 | 버전 (확인 환경) | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|---|
| faster-whisper | `>=1.2` (1.2.1) | 설치 — 음성 인식 | MIT | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 가능 | |
| CTranslate2 | (4.8.2, faster-whisper 의존) | 설치 | MIT | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) | 가능 | |
| PyAV (`av`) | (18.1.0, faster-whisper 의존) | 설치 | BSD-3-Clause — **휠에 FFmpeg(LGPL-3.0+)·libx264/libx265(GPL-2.0+) 등 공유 라이브러리 동봉** | [PyAV-Org/PyAV](https://github.com/PyAV-Org/PyAV) | 조건부 — [#1](https://github.com/gggg8657/meeting-local/issues/1) | 번들 재배포 시 고지·소스 제공 확인 |
| sherpa-onnx | `>=1.12` (1.13.8) | 설치 — 화자 분리 | Apache-2.0 | [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 가능 | |
| onnxruntime | (1.30.0, sherpa-onnx 의존) | 설치 | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | 가능 | |
| huggingface_hub | (1.33.0) | 설치 — Whisper 가중치 다운로드 | Apache-2.0 | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | 가능 | |
| 그 밖의 간접 의존성 | numpy (BSD-3), tokenizers (Apache-2.0), httpx/httpcore (BSD-3), anyio·h11·PyYAML·filelock (MIT), certifi·tqdm (MPL-2.0 포함), protobuf (BSD-3), flatbuffers·hf-xet·packaging (Apache-2.0 등) | 설치 | 각 패키지 메타데이터 | PyPI | 가능 | MPL-2.0 은 수정 시 해당 파일 공개 조건 |
| NVIDIA cuBLAS / cuDNN / NVRTC (`nvidia-*-cu12`) | (12.9 / 9.27) | 설치 — GPU 가 있을 때만 `setup.sh` 가 추가 설치 | NVIDIA 독점 (LicenseRef-NVIDIA-Proprietary) | [NVIDIA 소프트웨어 라이선스](https://docs.nvidia.com/cuda/eula/) | 조건부 — [#1](https://github.com/gggg8657/meeting-local/issues/1) | 재배포 범위 확인. `WHISPER_DEVICE=cpu` 면 불필요 |

## 내려받는 가중치 (저장소에 없음)

| 이름 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|
| Whisper CT2 변환 `Systran/faster-whisper-large-v3` (기본, `WHISPER_MODEL` 로 small/medium 등) | 음성 인식 | MIT (HF 모델 카드) | [HF](https://huggingface.co/Systran/faster-whisper-large-v3) · 원 모델 [openai/whisper](https://github.com/openai/whisper) (MIT) | 가능 | `models/whisper` 에 캐시 |
| `sherpa-onnx-pyannote-segmentation-3-0` | 화자 분할 | MIT (Copyright (c) 2022 CNRS, 압축 안 `LICENSE` 포함) | [k2-fsa 릴리스](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models) · 원 모델 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | 가능 | 원 HF 저장소는 게이트(연락처 제공 동의)가 있으나 k2-fsa 의 ONNX 변환본은 GitHub 릴리스로 받아 HF 토큰·동의 불필요 |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 화자 임베딩 | Apache-2.0 (ModelScope 카드) | [k2-fsa 릴리스](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) · 원 모델 [iic/speech_campplus_sv_zh_en_16k-common_advanced](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced) · [3D-Speaker](https://github.com/modelscope/3D-Speaker) (Apache-2.0) | 가능 | |

## 외부 프로그램·서비스 (동봉 안 함)

| 이름 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|
| ffmpeg / ffprobe | 녹음을 16 kHz WAV 로 변환·자르기 (`app.py`, 별도 프로세스) | LGPL-2.1+ / GPL (빌드에 따라) | [ffmpeg.org/legal](https://ffmpeg.org/legal.html) | 가능 | 시스템 설치본 사용 |
| kordoc (선택) | 회의록 HWPX 생성·검증 (`node` 로 호출) | MIT (Copyright (c) 2026 chrisryugj) | [chrisryugj/kordoc](https://github.com/chrisryugj/kordoc) | 가능 | 옆의 kordoc-local 설치본 사용 |
| nvidia-smi | GPU 고르기 (`gpu_pick.py`) | NVIDIA 드라이버 구성요소 | — | 가능 | 시스템 설치본 |
| Ollama | LLM 서버 | MIT | [ollama/ollama](https://github.com/ollama/ollama) | 가능 | `setup.sh` 는 서버가 없으면 공식 설치 스크립트로 설치 |
| Google Gemma 4 `gemma4:31b` (포털 실행 시) / Qwen3 `qwen3:8b` (`setup.sh` 기본 `MODEL`) | 회의록·인사이트 생성 | Apache-2.0 / Apache-2.0 | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) · [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 가능 | 가중치 동봉 안 함 |

## 확인 필요

- [#1 PyAV 휠 동봉 FFmpeg·x264/x265 (LGPL-3.0+/GPL-2.0+) · NVIDIA cuBLAS/cuDNN (독점) — pack.sh 번들](https://github.com/gggg8657/meeting-local/issues/1)
  - 내부 설치·사용은 해당 없음. `pack.sh` 번들을 **외부에 넘길 때** 고지·소스 제공·NVIDIA 재배포 조건 확인.
  - 회피: 번들을 넘기지 말고 받는 쪽이 `setup.sh` 로 직접 내려받게 함(기본 경로). NVIDIA 휠을 피하려면 `WHISPER_DEVICE=cpu`.
