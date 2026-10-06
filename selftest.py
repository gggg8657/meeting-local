#!/usr/bin/env python3
"""LLM·ASR·화자분리 없이 결정적 부분만 검증: 병합 → 화자 요약 → 이름 치환 → 주제 분할(폴백) → 인사이트/회의록 조립.
무거운 모델은 가짜로 바꿔 끼운다.   python3 selftest.py"""
import os, shutil, subprocess, app

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
# 저작권 표기: 서버가 화면에 붙이는 코드가 있어야 한다 (LICENSE·NOTICE)
_src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "app.py"), encoding="utf-8").read()
assert "wqkgMjAyNiDquYDrj5nso7wgwrcgZG9uZ2p1a2ltLmRldkBnbWFpbC5jb20=" in _src and "signed(" in _src and "X-Author" in _src, "저작권 표기 누락"

print("selftest OK — LLM calls:", len(calls))
