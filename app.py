#!/usr/bin/env python3
"""meeting local — 회의 녹음 → 화자 분리 전사 → 화자 이름 지정 → 회의록·인사이트 (로컬 LLM).

  python3 app.py                                  # http://localhost:8767
  python3 app.py --cli 회의.m4a [--speakers 3] [--names "김부장,이과장,박연구원"]

파이프라인:
  ffmpeg (→16kHz mono wav)
  → faster-whisper (ASR, ko)            [WHISPER_MODEL, 기본 large-v3 / 맥 PoC small]
  → sherpa-onnx (pyannote seg + 3D-Speaker 임베딩, 화자 분리, HF 토큰 불필요)
  → 병합 [{start,end,speaker,text}]   → 화자 이름 지정(UI)
  → LLM 1콜 주제 분할 → 청크별 인사이트 1콜씩 → 회의록 1콜 (kordoc 회의록 프리셋 Markdown)
  → (선택) kordoc generate --preset 회의록 → HWPX
"""
import base64
import datetime
import json
import os
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gpu_pick import Lazy, label, pick, torch_device

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("WORKSPACE") or os.path.join(ROOT, "_workspace")  # 포털이 AGENT_DATA/<도구> 로 모아 줌
MODELS = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
LLM_API = os.environ.get("LLM_API", "ollama")
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1" if LLM_API == "openai" else "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
NUM_CTX = int(os.environ.get("NUM_CTX", "16384"))
PORT = int(os.environ.get("PORT", "8767"))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")     # auto|cpu|cuda
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "")       # 비우면 cuda→float16, cpu→int8
CHUNK_SEC = int(os.environ.get("CHUNK_SEC", "600"))            # 주제 분할 실패 시 10분 창
KORDOC = os.path.join(ROOT, "..", "kordoc-local", "node_modules", "kordoc", "dist", "cli.js")
EXTS = (".m4a", ".mp3", ".wav", ".mp4", ".webm", ".ogg", ".flac", ".aac", ".mov")
SEG_MODEL = os.path.join(MODELS, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB_MODEL = os.path.join(MODELS, "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def jdump(p, o):
    write(p, json.dumps(o, ensure_ascii=False, indent=1))


# ── LLM (kordoc-local 과 동일) ───────────────────────────────────────────
def _clean(out):
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    out = re.sub(r"^```\w*\s*\n", "", out)
    out = re.sub(r"\n?```\s*$", "", out)
    return out.strip()


def openai_chat(system, user, model, on_token=None):
    body = {"model": model, "stream": True, "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {})}
    req = urllib.request.Request(LLM_BASE + "/chat/completions", json.dumps(body).encode(), hdr)
    buf = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                tok = (json.loads(line[5:])["choices"][0].get("delta") or {}).get("content") or ""
                if tok:
                    buf.append(tok)
                    if on_token:
                        on_token(tok)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    return "".join(buf)


def ollama(system, user, model, on_token=None):
    if LLM_API == "openai":
        return openai_chat(system, user, model, on_token)
    body = {"model": model, "stream": True, "think": False,
            "options": {"temperature": 0.2, "num_ctx": NUM_CTX},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in (0, 1):
        try:
            req = urllib.request.Request(LLM_BASE + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
            buf = []
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    if "error" in j:
                        raise RuntimeError(j["error"])
                    tok = j.get("message", {}).get("content", "")
                    if tok:
                        buf.append(tok)
                        if on_token:
                            on_token(tok)
                    if j.get("done"):
                        break
            return "".join(buf)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if attempt == 0 and "think" in msg:
                body.pop("think")
                continue
            raise RuntimeError(f"Ollama HTTP {e.code}: {msg[:300]}")


def models():
    if LLM_API == "openai":
        req = urllib.request.Request(LLM_BASE + "/models", headers={"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {})
        with urllib.request.urlopen(req, timeout=10) as r:
            return [m["id"] for m in json.load(r)["data"]]
    with urllib.request.urlopen(LLM_BASE + "/api/tags", timeout=10) as r:
        return [m["name"] for m in json.load(r)["models"]]


def prompts():
    txt = read(os.path.join(ROOT, "goal-prompt.md"))
    common, *rest = re.split(r"^## (\w+)\s*$", txt, flags=re.M)
    return common.strip(), {rest[i]: rest[i + 1].strip() for i in range(0, len(rest), 2)}


# ── 오디오 · ASR · 화자분리 ─────────────────────────────────────────────
def to_wav(src, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst], check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", dst], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def clip(wav, start, end, dst):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", wav, dst], check=True)


def _cuda_libs():
    """pip 으로 넣은 CUDA 12 cuBLAS·cuDNN(nvidia-*-cu12)을 미리 올린다 — ctranslate2 는 시스템 라이브러리 경로에서만 찾는다"""
    import ctypes, glob, site
    for sp in site.getsitepackages():
        for pat in ("nvidia/cublas/lib/libcublasLt.so.*", "nvidia/cublas/lib/libcublas.so.*", "nvidia/cudnn/lib/libcudnn*.so.*"):
            for f in sorted(glob.glob(os.path.join(sp, pat))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


WHISPER_WHERE = "아직 안 올림"


def _load_whisper():
    """GPU 는 고정하지 않는다 — 올릴 때마다 여유 메모리가 가장 큰 GPU 1장(없으면 CPU)"""
    global WHISPER_WHERE
    _cuda_libs()
    from faster_whisper import WhisperModel
    dev = WHISPER_DEVICE
    if dev == "auto":
        try:
            import ctranslate2
            dev = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            dev = "cpu"
    idx = 0
    if dev == "cuda":
        g = pick(6000)                      # large-v3 float16 ≈ 3~5GB
        if g:
            idx = int(torch_device(g).split(":")[1]); WHISPER_WHERE = label(g)
        else:
            dev, WHISPER_WHERE = "cpu", label(None)
    else:
        WHISPER_WHERE = dev
    compute = WHISPER_COMPUTE or ("float16" if dev == "cuda" else "int8")
    print(f"[asr] faster-whisper {WHISPER_MODEL} {WHISPER_WHERE} 에서 로드 compute={compute}", flush=True)
    return WhisperModel(WHISPER_MODEL, device=dev, device_index=idx, compute_type=compute, download_root=os.path.join(MODELS, "whisper"),
                        local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")))  # 폐쇄망: HF 핑 없이 로컬 캐시만


_WHISPER = Lazy(_load_whisper, "faster-whisper", log=lambda s: print(s, flush=True))  # 처음 쓸 때 올리고, 오래 안 쓰면 내림(GPU_IDLE_UNLOAD_S)


def transcribe(wav, log):
    with _WHISPER.use() as m:
        log.append(f"[asr] faster-whisper {WHISPER_MODEL} — {WHISPER_WHERE}")
        segs, info = m.transcribe(wav, language="ko", vad_filter=True, beam_size=5, word_timestamps=True)
        out = []
        for s in segs:  # segs 는 생성기 — 실제 계산이 여기서 돌므로 use() 안에서 다 꺼낸다
            words = [{"s": round(w.start, 2), "e": round(w.end, 2), "w": w.word} for w in (s.words or [])]
            out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "words": words})
    log.append(f"[asr] {len(out)} segments, lang={info.language} p={info.language_probability:.2f}")
    return out


def diarize(wav, num_speakers, log):
    import sherpa_onnx, wave, array
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=float(os.environ.get("DIAR_THRESHOLD", "0.5"))),
        min_duration_on=0.3, min_duration_off=0.5)
    if not cfg.validate():
        raise RuntimeError(f"화자분리 모델 없음: {SEG_MODEL}, {EMB_MODEL}")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    w = wave.open(wav)
    a = array.array("h"); a.frombytes(w.readframes(w.getnframes()))
    samples = [x / 32768.0 for x in a]
    turns = [{"start": round(s.start, 2), "end": round(s.end, 2), "spk": s.speaker} for s in sd.process(samples).sort_by_start_time()]
    log.append(f"[diar] {len(turns)} turns, speakers={sorted({t['spk'] for t in turns})}")
    return turns


def merge(segments, turns):
    """ASR 세그먼트마다 겹침 시간이 가장 긴 화자를 배정. 단어 타임스탬프가 있으면 단어 단위로 화자 전환을 쪼갠다.
    화자 라벨은 등장 순서로 화자1,2,3…"""
    def spk_at(s, e):
        best, bt = None, 0.0
        for t in turns:
            ov = min(e, t["end"]) - max(s, t["start"])
            if ov > bt:
                best, bt = t["spk"], ov
        return best
    order, out = {}, []
    def label(k):
        if k is None:
            return "화자?"
        if k not in order:
            order[k] = len(order) + 1
        return f"화자{order[k]}"
    for seg in segments:
        words = seg.get("words") or []
        if len(words) >= 4 and turns:
            cur, buf, s0 = None, [], None
            for w in words:
                k = spk_at(w["s"], w["e"]) if w["e"] > w["s"] else cur
                if k is None:
                    k = cur
                if cur is not None and k != cur and buf:
                    out.append({"start": s0, "end": buf[-1]["e"], "speaker": label(cur), "text": "".join(x["w"] for x in buf).strip()})
                    buf, s0 = [], None
                if s0 is None:
                    s0 = w["s"]
                buf.append(w); cur = k
            if buf:
                out.append({"start": s0, "end": buf[-1]["e"], "speaker": label(cur), "text": "".join(x["w"] for x in buf).strip()})
        else:
            out.append({"start": seg["start"], "end": seg["end"], "speaker": label(spk_at(seg["start"], seg["end"])), "text": seg["text"]})
    # 같은 화자 연속 발화 합치기 (2초 이내 간격)
    merged = []
    for u in out:
        if merged and merged[-1]["speaker"] == u["speaker"] and u["start"] - merged[-1]["end"] < 2.0:
            merged[-1]["end"] = u["end"]; merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
        else:
            merged.append(dict(u))
    return [u for u in merged if u["text"]]


def speakers_of(utts):
    """화자별 발화량·대표 발췌 2~3개 (이름 지정 UI용)."""
    info = {}
    for u in utts:
        d = info.setdefault(u["speaker"], {"name": "", "secs": 0.0, "n": 0, "samples": []})
        d["secs"] += u["end"] - u["start"]; d["n"] += 1
    for spk, d in info.items():
        cand = sorted((u for u in utts if u["speaker"] == spk), key=lambda u: -(u["end"] - u["start"]))[:3]
        d["samples"] = [{"start": u["start"], "end": min(u["end"], u["start"] + 8), "text": u["text"][:80]} for u in sorted(cand, key=lambda u: u["start"])]
        d["secs"] = round(d["secs"], 1)
    return dict(sorted(info.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 99)))


def rename(utts, names):
    """names: {"화자1": "김부장", ...}. 빈 값은 그대로."""
    return [dict(u, speaker=names.get(u["speaker"]) or u["speaker"]) for u in utts]


def ts(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def transcript_text(utts):
    return "\n".join(f"[{ts(u['start'])}] {u['speaker']}: {u['text']}" for u in utts)


# ── 회의록 · 인사이트 ────────────────────────────────────────────────────
def chunk(utts, llm, log):
    """LLM 주제 분할 → [{title, start, end}] ; 실패/빈 응답이면 CHUNK_SEC 창."""
    common, roles = prompts()
    lines = "\n".join(f"{i}|{ts(u['start'])}|{u['speaker']}|{u['text'][:60]}" for i, u in enumerate(utts))
    chunks = []
    if len(utts) >= 6:
        try:
            raw = _clean(llm(common + "\n\n" + roles["segment"], lines, MODEL))
            j = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
            idx = sorted({int(x["start_index"]) for x in j if 0 <= int(x["start_index"]) < len(utts)} | {0})
            titles = {int(x["start_index"]): str(x.get("title", "")) for x in j}
            for a, b in zip(idx, idx[1:] + [len(utts)]):
                chunks.append({"title": titles.get(a, ""), "i0": a, "i1": b})
            log.append(f"[segment] LLM 주제 {len(chunks)}개")
        except Exception as e:
            log.append(f"[segment] LLM 분할 실패 → 시간 창 ({type(e).__name__}: {e})")
            chunks = []
    if not chunks:
        i0, t0 = 0, utts[0]["start"] if utts else 0
        for i, u in enumerate(utts):
            if u["start"] - t0 >= CHUNK_SEC:
                chunks.append({"title": "", "i0": i0, "i1": i}); i0, t0 = i, u["start"]
        chunks.append({"title": "", "i0": i0, "i1": len(utts)})
    for c in chunks:
        part = utts[c["i0"]:c["i1"]]
        c["start"], c["end"] = part[0]["start"], part[-1]["end"]
        c["speakers"] = sorted({u["speaker"] for u in part})
        c["title"] = c["title"] or f"{ts(c['start'])}~{ts(c['end'])}"
    return chunks


def insights(utts, chunks, llm, log, emit):
    common, roles = prompts()
    out = []
    for n, c in enumerate(chunks, 1):
        emit({"stage": "insight", "msg": f"인사이트 {n}/{len(chunks)} — {c['title']}"})
        part = transcript_text(utts[c["i0"]:c["i1"]])
        txt = _clean(llm(common + "\n\n" + roles["insight"], f"[구간] {c['title']} ({ts(c['start'])}~{ts(c['end'])})\n[참여] {', '.join(c['speakers'])}\n\n{part}", MODEL))
        out.append({**{k: c[k] for k in ("title", "start", "end", "speakers")}, "insight": txt})
    return out


def minutes(meta, utts, ins, llm, emit):
    common, roles = prompts()
    emit({"stage": "minutes", "msg": "회의록 생성", "reset": True})
    user = (f"[회의 정보] 제목: {meta.get('title') or '(미정)'} / 일시: {meta.get('date') or '(미정)'} / 장소: {meta.get('place') or '(미정)'}\n"
            f"[참석자] {', '.join(sorted({u['speaker'] for u in utts}))}\n\n[구간별 인사이트]\n"
            + "\n".join(f"- {i['title']} ({ts(i['start'])}~{ts(i['end'])}): {i['insight'][:400]}" for i in ins)
            + "\n\n[전사본]\n" + transcript_text(utts))
    return _clean(llm(common + "\n\n" + roles["minutes"], user, MODEL, on_token=lambda t: emit({"token": t})))


def hwpx(md_path, out_path, log):
    if not os.path.exists(KORDOC):
        return None
    r = subprocess.run(["node", KORDOC, "generate", md_path, "-o", out_path, "--preset", "회의록"], capture_output=True, text=True)
    log.append("[kordoc] " + (r.stdout + r.stderr).strip()[-500:])
    v = subprocess.run(["node", KORDOC, "validate", out_path], capture_output=True, text=True)
    log.append("[kordoc validate] " + (v.stdout + v.stderr).strip()[-200:])
    return os.path.basename(out_path) if v.returncode == 0 else None


# ── 파이프라인 단계 (UI 는 두 단계: transcribe → finalize) ───────────────
def run_transcribe(src, num_speakers=0, emit=lambda ev: None, run_id=None):
    if not src.lower().endswith(EXTS):
        raise ValueError(f"지원하지 않는 형식: {os.path.basename(src)} (지원: {', '.join(EXTS)})")
    run_id = run_id or f"{datetime.date.today()}-{secrets.token_hex(2)}"
    d = os.path.join(WS, run_id); os.makedirs(d, exist_ok=True)
    log = []
    emit({"stage": "convert", "msg": "ffmpeg 16kHz 변환"})
    wav = os.path.join(d, "audio.wav")
    dur = to_wav(src, wav); log.append(f"[ffmpeg] {dur:.1f}s")
    emit({"stage": "asr", "msg": f"전사 ({WHISPER_MODEL})"})
    segs = transcribe(wav, log)
    emit({"stage": "diar", "msg": "화자 분리"})
    turns = diarize(wav, num_speakers, log)
    utts = merge(segs, turns)
    spk = speakers_of(utts)
    jdump(os.path.join(d, "transcript.json"), utts); jdump(os.path.join(d, "speakers.json"), spk)
    state = {"run_id": run_id, "file": os.path.basename(src), "duration": round(dur, 1), "ts": datetime.datetime.now().isoformat(timespec="seconds"),
             "n_speakers": len(spk), "status": "transcribed", "log": "\n".join(log)}
    jdump(os.path.join(d, "state.json"), state)
    return {**state, "utts": utts, "speakers": spk}


def run_finalize(run_id, names, meta, emit=lambda ev: None, llm=ollama):
    d = os.path.join(WS, run_id)
    utts = rename(jload(os.path.join(d, "transcript.json")), names or {})
    spk = jload(os.path.join(d, "speakers.json"))
    for k, v in (names or {}).items():
        if k in spk:
            spk[k]["name"] = v
    jdump(os.path.join(d, "speakers.json"), spk)
    state = jload(os.path.join(d, "state.json")); log = [state.get("log", "")]
    emit({"stage": "segment", "msg": "주제 분할"})
    chunks = chunk(utts, llm, log)
    ins = insights(utts, chunks, llm, log, emit)
    md = minutes(meta or {}, utts, ins, llm, emit)
    write(os.path.join(d, "minutes.md"), md); jdump(os.path.join(d, "insights.json"), ins)
    write(os.path.join(d, "transcript.txt"), transcript_text(utts))
    hw = hwpx(os.path.join(d, "minutes.md"), os.path.join(d, "minutes.hwpx"), log)
    state.update({"status": "done", "names": names or {}, "meta": meta or {}, "hwpx": hw, "log": "\n".join(log)})
    jdump(os.path.join(d, "state.json"), state)
    return {**state, "utts": utts, "speakers": spk, "minutes": md, "insights": ins}


def load_run(run_id):
    d = os.path.join(WS, run_id)
    st = jload(os.path.join(d, "state.json"))
    utts = rename(jload(os.path.join(d, "transcript.json")), st.get("names") or {})
    r = {**st, "utts": utts, "speakers": jload(os.path.join(d, "speakers.json"))}
    if os.path.exists(os.path.join(d, "minutes.md")):
        r["minutes"] = read(os.path.join(d, "minutes.md")); r["insights"] = jload(os.path.join(d, "insights.json"))
    return r


def list_runs():
    out = []
    if not os.path.isdir(WS):
        return out
    for name in sorted(os.listdir(WS), key=lambda n: os.path.getmtime(os.path.join(WS, n)), reverse=True)[:50]:
        p = os.path.join(WS, name, "state.json")
        if os.path.exists(p):
            try:
                j = jload(p); out.append({k: j.get(k) for k in ("run_id", "file", "duration", "n_speakers", "status", "ts")})
            except Exception:
                pass
    return out


# ── HTTP ───────────────────────────────────────────────────────────────
HTML = read(os.path.join(ROOT, "ui.html")) if os.path.exists(os.path.join(ROOT, "ui.html")) else "ui.html 없음"
RUN_RE = r"\d{4}-\d{2}-\d{2}-[0-9a-f]{4}"


# ── 윤문하기: kordoc-local 의 글 윤문 API (숫자·날짜·고유 표기가 바뀐 조각은 원문 유지) ─────────
KORDOC_URL = os.environ.get("KORDOC_URL", "http://localhost:8766").rstrip("/")


def polish_remote(text, strength="standard"):
    req = urllib.request.Request(KORDOC_URL + "/api/polish_text", json.dumps({"text": text, "strength": strength}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(json.loads(e.read() or b"{}").get("error") or f"kordoc HTTP {e.code}")


def polish_run(run_id, strength):
    """저장된 회의록(minutes.md)을 윤문해 덮어쓰고(처음 원문은 minutes_orig.md 로 보관) HWPX 를 다시 만든다"""
    if not re.fullmatch(RUN_RE, run_id or ""): raise ValueError("잘못된 실행 ID")
    d = os.path.join(WS, run_id); md = os.path.join(d, "minutes.md")
    if not os.path.exists(md): raise ValueError("회의록이 아직 없습니다")
    r = polish_remote(read(md), strength)
    if not os.path.exists(os.path.join(d, "minutes_orig.md")): write(os.path.join(d, "minutes_orig.md"), read(md))
    write(md, r["output"])
    log = []
    r["hwpx"] = hwpx(md, os.path.join(d, "minutes.hwpx"), log) if os.path.exists(os.path.join(d, "minutes.hwpx")) or os.path.exists(KORDOC) else None
    r["log"] = (r.get("log") or "") + "\n".join(log)
    return r


def polish_ok():
    try:
        urllib.request.urlopen(KORDOC_URL + "/api/models", timeout=2)
        return True
    except Exception:
        return False

# ── 저작권 표기 (LICENSE·NOTICE 참고) ─────────────────────────────────────
_SIG = __import__("base64").b64decode("wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==").decode()
_SIG_A = __import__("base64").b64decode("Z2dnZzg2NTcgPGRvbmdqdWtpbS5kZXZAZ21haWwuY29tPg==").decode()


def signed(html):
    """화면에 저작권 표기를 붙인다. ui.html 에서 지워져도 서버가 내보낼 때 다시 붙는다."""
    name, mail = _SIG.split(" · ")
    if 'name="author"' not in html:
        meta = f'<meta name="author" content="{name[7:]} <{mail}>">'
        html = html.replace("<head>", "<head>" + meta, 1) if "<head>" in html else meta + html
    if "data-sig" not in html:
        tag = (f'<!-- {_SIG} --><div data-sig title="{mail}" style="text-align:center;font-size:11px;color:#9aa0a6;'
               f'opacity:.55;margin:28px 0 8px">{name}</div>')
        html = html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag
    return html


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        if "/api/run" in (str(a[0]) if a else "") or "/api/finalize" in (str(a[0]) if a else ""):
            super().log_message(fmt, *a)

    def _send(self, body, ctype="application/json", code=200, name=None):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("X-Author", _SIG_A); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        if name:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{urllib.request.quote(name)}")
        self.end_headers(); self.wfile.write(b)

    def _sse(self):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
        def emit(ev):
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()
        return emit

    def do_GET(self):
        try:
            if self.path == "/api/polish_ok":
                return self._send({"ok": polish_ok()})
            if self.path == "/api/models":
                return self._send(models())
            if self.path == "/api/runs":
                return self._send(list_runs())
            m = re.fullmatch(rf"/api/runs/({RUN_RE})", self.path)
            if m:
                return self._send(load_run(m.group(1)))
            m = re.fullmatch(rf"/api/runs/({RUN_RE})/clip\?s=([\d.]+)&e=([\d.]+)", self.path)
            if m:  # 화자 확인용 짧은 클립
                d = os.path.join(WS, m.group(1)); s, e = float(m.group(2)), min(float(m.group(3)), float(m.group(2)) + 10)
                dst = os.path.join(d, f"clip_{s:.1f}_{e:.1f}.wav")
                os.path.exists(dst) or clip(os.path.join(d, "audio.wav"), s, e, dst)
                with open(dst, "rb") as f:
                    return self._send(f.read(), "audio/wav")
            m = re.fullmatch(rf"/api/runs/({RUN_RE})/(minutes\.md|minutes\.hwpx|insights\.json|transcript\.txt|transcript\.json)", self.path)
            if m:
                with open(os.path.join(WS, m.group(1), m.group(2)), "rb") as f:
                    return self._send(f.read(), "application/octet-stream", name=f"{m.group(1)}_{m.group(2)}")
            self._send(signed(HTML.replace("%MODEL%", json.dumps(MODEL))).encode(), "text/html; charset=utf-8")
        except FileNotFoundError:
            self._send({"error": "없음"}, code=404)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, code=500)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/api/polish":
            try:
                req = json.loads(raw or b"{}")
                return self._send(polish_run(req.get("run_id"), req.get("strength") or "standard"))
            except ValueError as e:
                return self._send({"error": str(e)}, code=400)
            except Exception as e:
                return self._send({"error": f"{type(e).__name__}: {e}"}, code=502)
        if self.path.split("?")[0] == "/v1/audio/transcriptions":  # OpenAI 호환 STT: multipart file 또는 JSON {"file": base64}
            try:
                ctype = self.headers.get("Content-Type", "")
                if ctype.startswith("multipart/form-data"):
                    import email.parser
                    msg = email.parser.BytesParser().parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
                    audio = next((part.get_payload(decode=True) for part in msg.walk() if part.get_param("name", header="content-disposition") == "file"), None)
                else:
                    audio = base64.b64decode(json.loads(raw)["file"])
                if not audio:
                    raise ValueError("file 없음")
                d = os.path.join(WS, "_stt"); os.makedirs(d, exist_ok=True)
                src = os.path.join(d, secrets.token_hex(4)); wav = src + ".wav"
                with open(src, "wb") as f:
                    f.write(audio)
                to_wav(src, wav)
                text = " ".join(s["text"] for s in transcribe(wav, [])).strip()
                for f in (src, wav):
                    os.remove(f)
                return self._send({"text": text})
            except Exception as e:
                return self._send({"error": f"{type(e).__name__}: {e}"}, code=500)
        req = json.loads(raw)
        try:
            if self.path == "/api/run":
                name = re.sub(r"[^\w.\-가-힣 ]", "_", os.path.basename(req.get("file_name") or "upload.bin"))
                run_id = f"{datetime.date.today()}-{secrets.token_hex(2)}"
                d = os.path.join(WS, run_id); os.makedirs(d)
                src = os.path.join(d, "00_" + name)
                with open(src, "wb") as f:
                    f.write(base64.b64decode(req["file_b64"]))
                emit = self._sse()
                try:
                    emit({"done": run_transcribe(src, int(req.get("speakers") or 0), emit, run_id)})
                except Exception as e:
                    emit({"error": f"{type(e).__name__}: {e}"})
                return
            m = re.fullmatch(rf"/api/finalize/({RUN_RE})", self.path)
            if m:
                emit = self._sse()
                try:
                    emit({"done": run_finalize(m.group(1), req.get("names") or {}, req.get("meta") or {}, emit)})
                except Exception as e:
                    emit({"error": f"{type(e).__name__}: {e}"})
                return
            self._send({"error": "unknown"}, code=404)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, code=500)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("audio"); ap.add_argument("--speakers", type=int, default=0)
        ap.add_argument("--names", default=""); ap.add_argument("--title", default=""); ap.add_argument("--date", default="")
        a = ap.parse_args(sys.argv[2:])
        pr = lambda ev: print(f"[{ev['stage']}] {ev['msg']}", file=sys.stderr) if "stage" in ev else None
        r = run_transcribe(a.audio, a.speakers, pr)
        names = {f"화자{i + 1}": n.strip() for i, n in enumerate(a.names.split(",")) if n.strip()} if a.names else {}
        r = run_finalize(r["run_id"], names, {"title": a.title, "date": a.date}, pr)
        print(r["minutes"]); print("\n--- 인사이트 ---", file=sys.stderr)
        for i in r["insights"]:
            print(f"[{ts(i['start'])}] {i['title']}: {i['insight'][:200]}", file=sys.stderr)
        print(f"\n[run] _workspace/{r['run_id']}  hwpx={r['hwpx']}", file=sys.stderr)
        sys.exit(0)
    print(f"meeting local → http://localhost:{PORT}  (llm={LLM_API} {LLM_BASE} {MODEL}, whisper={WHISPER_MODEL})  {_SIG}")
    ThreadingHTTPServer(("", PORT), H).serve_forever()
