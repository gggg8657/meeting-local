"""GPU 고르기 — 고정 배정 없이 '지금' 여유 메모리가 가장 큰 GPU 1장을 쓴다. stdlib 만 (도구마다 같은 파일을 복사해 둠).

  g = pick(8000)            # 여유 8GB 이상인 GPU 중 가장 넉넉한 것 → {"index": "2", "free": 81000, "util": 0} 또는 None(→ CPU 로)
  torch_device(g)           # 이 프로세스 안에서 쓸 이름 "cuda:N" (CUDA_VISIBLE_DEVICES 안에서의 순번)
  env_for(g, os.environ)    # 서브프로세스용: CUDA_VISIBLE_DEVICES=<그 번호> → 안에서는 cuda(:0)
  Lazy(load, "이름")         # 처음 쓸 때 올리고, GPU_IDLE_UNLOAD_S(기본 600초) 안 쓰면 내려서 VRAM 을 돌려준다 → 다음엔 다시 고름

env: GPU_POOL(쓸 GPU 번호 목록, 비우면 CUDA_VISIBLE_DEVICES, 그것도 비우면 전부)  GPU_IDLE_UNLOAD_S(600, 0 이면 안 내림)"""
import gc, os, subprocess, sys, threading, time
from contextlib import contextmanager

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch·ctranslate2 의 번호를 nvidia-smi 번호 순서와 맞춘다 (CUDA 초기화 전에)
IDLE_S = float(os.environ.get("GPU_IDLE_UNLOAD_S", "600"))


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


def pick(min_free_mb=0):
    """여유 메모리가 가장 큰 GPU (같으면 덜 바쁜 쪽). min_free_mb 를 넘는 게 없으면 None"""
    ok = [g for g in gpus() if g["free"] >= min_free_mb]
    return max(ok, key=lambda g: (g["free"] // 1024, -g["util"])) if ok else None


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
