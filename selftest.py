#!/usr/bin/env python3
"""LLM·ASR·화자분리 없이 결정적 부분만 검증: 병합 → 화자 요약 → 이름 치환 → 주제 분할(폴백) → 인사이트/회의록 조립.
무거운 모델은 가짜로 바꿔 끼운다.   python3 selftest.py"""
import os, shutil, subprocess, app
for _k in ("PORTAL_JOBS_URL", "PORTAL_JOBS_TOKEN", "PORTAL_JOBS_TOOL"): os.environ.pop(_k, None)   # setup.sh 가 포털 환경에서 돌려도 시험 작업을 진짜 🔔 로 보내지 않게

SEGS = [  # ASR 가짜 출력 (단어 타임스탬프 포함)
    {"start": 0.0, "end": 4.0, "text": "안녕하세요 오늘 회의 시작합니다", "words": [{"s": 0.0, "e": 1.0, "w": "안녕하세요"}, {"s": 1.0, "e": 2.0, "w": " 오늘"}, {"s": 2.0, "e": 3.0, "w": " 회의"}, {"s": 3.0, "e": 4.0, "w": " 시작합니다"}]},
    {"start": 4.5, "end": 8.0, "text": "네 확인했습니다 벤치마크는 제가 하겠습니다", "words": [{"s": 4.5, "e": 5.0, "w": "네"}, {"s": 5.0, "e": 6.0, "w": " 확인했습니다"}, {"s": 6.0, "e": 7.0, "w": " 벤치마크는"}, {"s": 7.0, "e": 8.0, "w": " 제가 하겠습니다"}]},
    {"start": 8.5, "end": 12.0, "text": "좋습니다 금요일까지 부탁합니다", "words": []},
]
TURNS = [{"start": 0.0, "end": 4.2, "spk": 7}, {"start": 4.3, "end": 8.1, "spk": 2}, {"start": 8.4, "end": 12.0, "spk": 7}]

# 1) 병합: 등장 순서로 화자1/2, 겹침 최대 화자 배정
u = app.merge(SEGS, TURNS)
assert [x["speaker"] for x in u] == ["화자1", "화자2", "화자1"], u
assert u[1]["text"].startswith("네") and u[2]["text"].startswith("좋습니다")
# 단어 단위 분할: 한 세그먼트 안에서 화자가 바뀌면 쪼개진다
u2 = app.merge([SEGS[0]], [{"start": 0, "end": 2.0, "spk": 1}, {"start": 2.0, "end": 4.0, "spk": 2}])
assert len(u2) == 2 and u2[0]["text"] == "안녕하세요 오늘" and u2[1]["speaker"] == "화자2", u2
# 2) 화자 요약 + 이름 치환
sp = app.speakers_of(u)
assert list(sp) == ["화자1", "화자2"] and sp["화자1"]["n"] == 2 and sp["화자1"]["samples"]
r = app.rename(u, {"화자1": "김부장"})
assert r[0]["speaker"] == "김부장" and r[1]["speaker"] == "화자2"
# 3) 주제 분할 폴백(짧으면 LLM 호출 없이 1청크) + 인사이트/회의록 조립 (가짜 LLM)
calls = []
def fake(system, user, model, on_token=None):
    calls.append(system.split("## ")[-1][:10] if "## " in system else system[:10])
    if "주제 단위" in system: return '[{"start_index":0,"title":"오프닝"},{"start_index":1,"title":"벤치마크"}]'
    if "인사이트 하나" in system: return "□ 핵심: 벤치마크 담당 확정\nㅇ 근거: 화자2가 맡겠다고 함"
    out = "# 주간회의\n\n## 1. 회의 개요\n  - 참석자: 김부장, 화자2\n\n## 4. 액션아이템\n\n| 번호 | 할 일 | 담당 | 기한 |\n| --- | --- | --- | --- |\n| 1 | 벤치마크 | 화자2 | 금요일 |\n"
    if on_token: on_token(out)
    return out
log = []
ch = app.chunk(r, fake, log)
assert len(ch) == 1 and ch[0]["speakers"] == ["김부장", "화자2"], ch  # 발화 6개 미만 → LLM 분할 생략
ins = app.insights(r, ch, fake, log, lambda ev: None)
assert ins[0]["insight"].startswith("□ 핵심") and ins[0]["start"] == 0.0
md = app.minutes({"title": "주간회의"}, r, ins, fake, lambda ev: None)
assert md.startswith("# 주간회의") and "| 벤치마크 |" in md
# 긴 전사본이면 LLM 분할이 쓰인다
long = [dict(x, start=x["start"] + 20 * i, end=x["end"] + 20 * i) for i in range(3) for x in r]
ch2 = app.chunk(long, fake, log)
assert len(ch2) == 2 and ch2[1]["title"] == "벤치마크" and ch2[1]["i0"] == 1, ch2
# 4) 전체 run: ASR/화자분리/ffmpeg 가짜 → 저장·재열람·finalize 왕복
app.transcribe = lambda wav, log: SEGS
app.diarize = lambda wav, n, log: TURNS
app.to_wav = lambda src, dst: (open(dst, "wb").write(b"RIFF"), 12.0)[1]
app.hwpx = lambda md, out, log: None
src = os.path.join(app.WS, "_selftest.wav"); os.makedirs(app.WS, exist_ok=True); open(src, "wb").write(b"RIFF")
st = app.run_transcribe(src, 0)
assert st["n_speakers"] == 2 and st["status"] == "transcribed"
fin = app.run_finalize(st["run_id"], {"화자2": "박연구원"}, {"title": "주간회의"}, llm=fake)
assert fin["status"] == "done" and fin["utts"][1]["speaker"] == "박연구원" and fin["minutes"].startswith("# ")
again = app.load_run(st["run_id"])
assert again["minutes"] == fin["minutes"] and again["utts"][1]["speaker"] == "박연구원"
assert app.ts(125) == "02:05" and "[00:00] 김부장:" in app.transcript_text(r)
shutil.rmtree(os.path.join(app.WS, st["run_id"])); os.remove(src)
# 5) 브라우저가 떠나도(첫 이벤트 뒤 쓰기가 BrokenPipeError) 전사·회의록 작성은 끝까지 — 실행 목록에 남고 🔔 done, 서버 오류·트레이스백 없음
import base64, json, portal_jobs
from http.server import BaseHTTPRequestHandler
import io, sys, tempfile, threading, urllib.request
from http.server import ThreadingHTTPServer
class Cut:
    def __init__(s, f, n): s.f, s.n = f, n
    def write(s, b):
        s.n -= 1
        if s.n < 0: raise BrokenPipeError(32, "Broken pipe")
        return s.f.write(b)
    def flush(s): pass
    def __getattr__(s, k): return getattr(s.f, k)
class CutH(app.H):
    def setup(s): super().setup(); s.wfile = Cut(s.wfile, 2)  # 응답 머리 + 첫 이벤트까지만 닿음
ERRS = []
class Srv(ThreadingHTTPServer):
    def handle_error(s, req, addr): ERRS.append(sys.exc_info()[1])
def cut_post(path, body, user="alice"):
    srv = Srv(("127.0.0.1", 0), CutH); threading.Thread(target=srv.serve_forever, daemon=True).start()
    err0, sys.stderr = sys.stderr, io.StringIO()
    try:
        r = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}", json.dumps(body).encode(), {"Content-Type": "application/json", "X-Agent-User": user})
        got = urllib.request.urlopen(r).read().decode()
    finally:
        cap, sys.stderr = sys.stderr.getvalue(), err0
    srv.shutdown(); srv.server_close()
    assert got.count("data: ") == 1 and not ERRS and "Traceback" not in cap and cap.count("연결 끊김") == 1, (got, ERRS, cap)
    return got
GOT = []
class FakePortal(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        GOT.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
fp = ThreadingHTTPServer(("127.0.0.1", 0), FakePortal); threading.Thread(target=fp.serve_forever, daemon=True).start()
_ws, _fin, _env = app.WS, app.run_finalize, dict(os.environ)
os.environ.update(PORTAL_JOBS_URL=f"http://127.0.0.1:{fp.server_address[1]}/api/internal/jobs", PORTAL_JOBS_TOKEN="tok-test", PORTAL_JOBS_TOOL="meeting-local")
for _k in ("http_proxy", "HTTP_PROXY"): os.environ.pop(_k, None)
app.WS = tempfile.mkdtemp(prefix="meeting-cut-")
app.run_finalize = lambda *a, **k: _fin(*a, llm=fake, **k)  # 진짜 LLM 대신 가짜
try:
    cut_post("/api/run", {"file_name": "떠난회의.wav", "file_b64": base64.b64encode(b"RIFF").decode()})
    rid = next(x for x in os.listdir(os.path.join(app.WS, "u", "alice")) if x[:4].isdigit())
    assert app.load_run(rid, os.path.join(app.WS, "u", "alice"))["status"] == "transcribed"
    cut_post(f"/api/finalize/{rid}", {"names": {"화자2": "박연구원"}, "meta": {"title": "떠난 회의"}})
    fin2 = app.load_run(rid, os.path.join(app.WS, "u", "alice"))
    assert fin2["status"] == "done" and fin2["minutes"].startswith("# "), fin2.get("status")
    portal_jobs.flush()
    assert [(g["id"], g["status"]) for g in GOT] == [(rid, "running"), (rid, "done"), (rid + "-min", "running"), (rid + "-min", "done")], GOT
finally:
    fp.shutdown(); shutil.rmtree(app.WS, ignore_errors=True); app.WS, app.run_finalize = _ws, _fin
    os.environ.clear(); os.environ.update(_env)

# 저작권 표기: 서버가 화면에 붙이는 코드가 있어야 한다 (LICENSE·NOTICE)
_src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "app.py"), encoding="utf-8").read()
assert "wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==" in _src and "signed(" in _src and "X-Author" in _src, "저작권 표기 누락"

print("selftest OK — LLM calls:", len(calls))

# GPU 후보: 부모 후보가 0,1,2 면 3(운영 vLLM 전용)을 절대 고르지 않음 — nvidia-smi 출력은 가짜, 예약 장부는 임시 폴더
import gpu_pick as _gp, os as _os, tempfile as _tf, shutil as _sh
_smi0, _dir0, _env0 = _gp._smi, _gp.RES_DIR, dict(_os.environ)
try:
    _gp._smi = lambda: "0, 1000, 90\n1, 2000, 90\n2, 3000, 90\n3, 90000, 0\n"  # 3 이 가장 한가해 보이게
    _gp.RES_DIR = _tf.mkdtemp()
    for _k in ("GPU_POOL", "GPU_RESERVED", "VLLM_GPUS"): _os.environ.pop(_k, None)
    for _res in ("none", "3"):
        _os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2", GPU_RESERVED=_res)
        assert [g["index"] for g in _gp.gpus()] == ["0", "1", "2"]
        assert _gp.pick(0, hold_s=0)["index"] == "2" and _gp.torch_device(_gp.pick(0, hold_s=0)) == "cuda:2"
        assert _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == "0,1,2" == _gp.env_for(None, _os.environ)["CUDA_VISIBLE_DEVICES"]
    # 셸(~/.zshrc)에서 2,3 을 물려받아도 예약된 3 은 빼고 2 만 — 보이는 번호는 cuda:0
    _os.environ.update(CUDA_VISIBLE_DEVICES="2,3", GPU_RESERVED="3")
    assert _gp.pick(0, hold_s=0)["index"] == "2" and _gp.torch_device(_gp.pick(0, hold_s=0)) == "cuda:0"
    assert _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == "2"
    _os.environ.pop("GPU_RESERVED"); _os.environ["VLLM_GPUS"] = "3"  # GPU_RESERVED 없으면 VLLM_GPUS
    assert _gp.pick(0, hold_s=0)["index"] == "2"
    _os.environ["GPU_POOL"] = "0,1"  # 교집합이 비면 GPU 를 하나도 안 보임(CPU)
    assert _gp.pick(0, hold_s=0) is None and _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == ""
    if hasattr(app, "worker_env"):  # 작업자(그림·영상) 프로세스 환경
        assert app.worker_env({"CUDA_VISIBLE_DEVICES": "0,1,2", "GPU_RESERVED": "none"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "0,1,2"
        assert app.worker_env({"CUDA_VISIBLE_DEVICES": "2,3", "GPU_RESERVED": "3"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "2"
        assert app.worker_env({"GPU_POOL": "0,1,2,3", "VLLM_GPUS": "3"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "0,1,2"
finally:
    _sh.rmtree(_gp.RES_DIR, ignore_errors=True); _gp._smi, _gp.RES_DIR = _smi0, _dir0; _os.environ.clear(); _os.environ.update(_env0)
print("selftest GPU 후보 OK — 0,1,2 에서 3 안 고름")

# 포털 작업 알림(portal_jobs): 환경변수 없으면 False(아무 일 없음), 있으면 가짜 포털로 POST + 토큰 헤더
import json as _json, os as _os, threading as _th, portal_jobs as _pj
from http.server import BaseHTTPRequestHandler as _BH, HTTPServer as _HS
_env0 = dict(_os.environ)
try:
    for _k in ("PORTAL_JOBS_URL", "PORTAL_JOBS_TOKEN", "PORTAL_JOBS_TOOL"): _os.environ.pop(_k, None)
    assert _pj.notify("alice", "running", id="j1", title="t") is False and _pj.enabled() is False
    _got = []
    class _P(_BH):
        def log_message(self, *a): pass
        def do_POST(self):
            _got.append((self.headers.get("X-Portal-Jobs-Token"), _json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
    _srv = _HS(("127.0.0.1", 0), _P); _th.Thread(target=_srv.serve_forever, daemon=True).start()
    _os.environ.update(PORTAL_JOBS_URL=f"http://127.0.0.1:{_srv.server_address[1]}/api/internal/jobs", PORTAL_JOBS_TOKEN="tok-test", PORTAL_JOBS_TOOL="meeting-local")
    _os.environ.pop("http_proxy", None); _os.environ.pop("HTTP_PROXY", None)
    app.job(type("_H", (), {"headers": {"X-Agent-User": "alice"}})(), "done", "2026-10-14-abcd-min", "회의록 작성", "인사이트 2개")
    assert _pj.flush() and len(_got) == 1, _got
    _tok, _p = _got[0]
    assert _tok == "tok-test" and _p["tool"] == "meeting-local" and _p["user"] == "alice" and _p["status"] == "done", _p
    assert _p["id"] == "2026-10-14-abcd-min" and _p["link"] == "/?run=2026-10-14-abcd" and _p["message"] == "인사이트 2개", _p
    assert _pj.notify("alice", "bogus", id="x") is False
    _srv.shutdown()
finally:
    _os.environ.clear(); _os.environ.update(_env0)
print("selftest 포털 작업 알림 OK")
