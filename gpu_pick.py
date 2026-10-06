"""GPU 고르기 — 고정 배정 없이 '지금' 여유 메모리가 가장 큰 GPU 1장을 쓴다. stdlib 만 (도구마다 같은 파일을 복사해 둠).

  g = pick(8000)            # 여유 8GB 이상인 GPU 중 가장 넉넉한 것 → {"index": "2", "free": 81000, "util": 0} 또는 None(→ CPU 로)
  torch_device(g)           # 이 프로세스 안에서 쓸 이름 "cuda:N" (CUDA_VISIBLE_DEVICES 안에서의 순번)
  env_for(g, os.environ)    # 서브프로세스용: CUDA_VISIBLE_DEVICES=<그 번호> → 안에서는 cuda(:0)
  release(g)                # (선택) 다 올렸으면 예약을 바로 푼다 — 안 불러도 GPU_PICK_HOLD_S 뒤 저절로 풀림
  Lazy(load, "이름")         # 처음 쓸 때 올리고, GPU_IDLE_UNLOAD_S(기본 600초) 안 쓰면 내려서 VRAM 을 돌려준다 → 다음엔 다시 고름

예약: 두 작업이 거의 동시에 고르면 둘 다 아직 메모리를 안 잡아서 같은 GPU 로 몰린다. 그래서 pick(min_free_mb) 은 고른 GPU 에
min_free_mb 만큼 '예약'을 남기고(모든 도구가 같이 보는 파일, flock), 다른 pick 은 그만큼 덜 비었다고 본다. 예약 중 실제로
잡혀서 nvidia-smi 에 보이는 만큼은 빼고 센다(이중으로 안 셈).

env: GPU_POOL(쓸 GPU 번호 목록, 비우면 CUDA_VISIBLE_DEVICES, 그것도 비우면 전부)  GPU_IDLE_UNLOAD_S(600, 0 이면 안 내림)
     GPU_PICK_HOLD_S(예약 유지 초, 600)  GPU_PICK_DIR(예약 파일 자리, 기본 /tmp/gpu_pick-<uid>)"""
import fcntl, gc, json, os, subprocess, sys, tempfile, threading, time, uuid
from contextlib import contextmanager

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch·ctranslate2 의 번호를 nvidia-smi 번호 순서와 맞춘다 (CUDA 초기화 전에)
IDLE_S = float(os.environ.get("GPU_IDLE_UNLOAD_S", "600"))
HOLD_S = float(os.environ.get("GPU_PICK_HOLD_S", "600"))
RES_DIR = os.environ.get("GPU_PICK_DIR") or os.path.join(tempfile.gettempdir(), f"gpu_pick-{os.getuid()}")


def _ids(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def gpus():
    """nvidia-smi 로 [{"index", "free"(MiB), "util"(%)}] — 쓸 수 있는(pool·visible) GPU 만. nvidia-smi 없으면 []"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    vis, pool = _ids(os.environ.get("CUDA_VISIBLE_DEVICES")), _ids(os.environ.get("GPU_POOL"))
    rows = []
    for line in out.strip().splitlines():
        try:
            i, free, util = [x.strip() for x in line.split(",")]
            g = {"index": i, "free": int(float(free)), "util": int(float(util)) if util.replace(".", "").isdigit() else 0}
        except ValueError:
            continue
        if (not vis or i in vis) and (not pool or i in pool):
            rows.append(g)
    return rows


@contextmanager
def _book():
    """예약 장부(json)를 잠그고 연다 — 만료된 예약은 버린다. 장부를 못 쓰면(권한 등) 빈 장부로 계속"""
    try:
        os.makedirs(RES_DIR, exist_ok=True)
        lockf = open(os.path.join(RES_DIR, "lock"), "a+")
    except OSError:
        yield []; return
    path = os.path.join(RES_DIR, "reservations.json")
    with lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            book = json.load(open(path))
        except (OSError, ValueError):
            book = []
        book[:] = [r for r in book if r.get("until", 0) > time.time()]
        yield book
        try:
            with open(path + ".tmp", "w") as f:
                json.dump(book, f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass


def pick(min_free_mb=0, hold_s=None):
    """여유 메모리가 가장 큰 GPU (같으면 덜 바쁜 쪽, 다른 작업의 예약은 뺌). min_free_mb 를 넘는 게 없으면 None.
    고르면 min_free_mb 만큼 hold_s(기본 GPU_PICK_HOLD_S)초 예약 — 바로 뒤에 고르는 작업이 같은 GPU 로 몰리지 않게"""
    hold_s = HOLD_S if hold_s is None else hold_s
    with _book() as book:
        rows = gpus()
        for g in rows:  # 예약분 중 아직 nvidia-smi 에 안 잡힌 만큼만 뺀다 (base = 예약할 때의 nvidia-smi 여유)
            g["raw"] = g["free"]
            g["free"] -= sum(max(0, r["mb"] - max(0, r["base"] - g["raw"])) for r in book if r["index"] == g["index"])
        ok = [g for g in rows if g["free"] >= min_free_mb]
        if not ok:
            return None
        g = max(ok, key=lambda g: (g["free"] // 1024, -g["util"]))
        if min_free_mb > 0 and hold_s > 0:
            g["rid"] = uuid.uuid4().hex[:12]
            book.append({"id": g["rid"], "index": g["index"], "mb": int(min_free_mb), "base": g["raw"], "until": time.time() + hold_s, "pid": os.getpid()})
        return g


def release(g):
    """pick 이 남긴 예약을 푼다 (모델을 다 올렸거나 실패했을 때). 안 불러도 시간이 지나면 풀림"""
    if g and g.get("rid"):
        with _book() as book:
            book[:] = [r for r in book if r.get("id") != g["rid"]]


def torch_device(g):
    """같은 프로세스 안에서: 물리 번호 → 'cuda:N' (CUDA_VISIBLE_DEVICES 가 있으면 그 안의 순번)"""
    if not g:
        return "cpu"
    vis = _ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return f"cuda:{vis.index(g['index']) if vis else int(g['index'])}"


def env_for(g, env):
    """서브프로세스용 환경: 그 GPU 1장만 보이게"""
    env = dict(env)
    if g:
        env.update(CUDA_VISIBLE_DEVICES=g["index"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    return env


def label(g):
    return f"GPU {g['index']} (여유 {g['free'] // 1024}GB)" if g else "CPU (여유 있는 GPU 없음)"


def free_cuda():
    gc.collect()
    if "torch" in sys.modules:
        try:
            sys.modules["torch"].cuda.empty_cache()
        except Exception:
            pass


class Lazy:
    """GPU 모델 하나를 '쓸 때만' 올려 둔다. load() 는 모델을 돌려주고(그 안에서 pick 으로 GPU 를 고름),
    아무도 안 쓴 지 IDLE_S 초가 지나면 내리고 VRAM 을 비운다. with m.use() as model: ..."""

    def __init__(self, load, name="model", idle_s=None, log=print, cleanup=None):
        self.load, self.name, self.idle_s, self.log = load, name, IDLE_S if idle_s is None else idle_s, log
        self.cleanup = cleanup  # 내릴 때 모델 밖에 붙들린 것(라이브러리 전역 캐시 등)도 풀어 주는 함수
        self.obj, self.busy, self.last = None, 0, time.time()
        self.lock = threading.Lock()
        self._reaper = None

    @contextmanager
    def use(self):
        with self.lock:
            self.busy += 1
        try:
            with self.lock:
                if self.obj is None:
                    self.obj = self.load()
                obj = self.obj
            if self.idle_s > 0 and not self._reaper:
                self._reaper = threading.Thread(target=self._reap, daemon=True); self._reaper.start()
            yield obj
        finally:
            with self.lock:
                self.busy -= 1; self.last = time.time()

    def unload(self):
        with self.lock:
            if self.obj is None or self.busy:
                return False
            self.obj = None
            if self.cleanup:
                self.cleanup()
        free_cuda()
        self.log(f"[gpu] {self.name} 내림 — {int(self.idle_s)}초 동안 안 써서 GPU 메모리 반환 (다음에 쓸 때 여유 많은 GPU 를 다시 고름)")
        return True

    def _reap(self):
        while True:
            time.sleep(min(30, max(1, self.idle_s / 4)))
            if self.obj is not None and not self.busy and time.time() - self.last > self.idle_s:
                self.unload()
