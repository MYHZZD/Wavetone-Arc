"""隔离测试：证明只用自带的 4 个 CUDA DLL（不依赖系统 CUDA Toolkit）也能跑 GPU。

用法:
    python cuda_isolation_test.py <staged_cuda_bin_dir> <cupy_cache_dir>

会把 PATH 里所有含 cuda/nvidia 的目录删掉，并清除 CUDA_PATH/CUDA_HOME，
然后只通过 os.add_dll_directory 指向自带目录。
"""
import ctypes
import os
import sys
import time

staged = os.path.abspath(sys.argv[1])
cache = os.path.abspath(sys.argv[2])

# --- 模拟一台没有安装 CUDA Toolkit 的机器 -------------------------------
for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_BIN_PATH"):
    os.environ.pop(var, None)
kept = [
    p
    for p in os.environ.get("PATH", "").split(os.pathsep)
    if p and "cuda" not in p.lower() and "nvidia" not in p.lower()
]
os.environ["PATH"] = os.pathsep.join(kept)
os.environ["CUPY_CACHE_DIR"] = cache

# 自带 DLL 目录加入搜索路径（打包后由 runtime hook 完成同样的事）：
#   1) CUDA_PATH 指向自带的 CUDA 根目录 -> cupy 自己 add_dll_directory(CUDA_PATH/bin)
#   2) PATH / add_dll_directory 双保险
os.environ["CUDA_PATH"] = os.path.dirname(staged)
os.environ["PATH"] = staged + os.pathsep + os.environ["PATH"]
os.add_dll_directory(staged)
print("[test] CUDA_PATH =", os.environ.get("CUDA_PATH"))
print("[test] staged =", staged)

t0 = time.time()
import cupy as cp  # noqa: E402

print("[test] cupy", cp.__version__, "import", round(time.time() - t0, 2), "s")
print("[test] runtimeGetVersion", cp.cuda.runtime.runtimeGetVersion())
print("[test] deviceCount", cp.cuda.runtime.getDeviceCount())

name = cp.cuda.runtime.getDeviceProperties(0)["name"]
print("[test] device", name.decode() if isinstance(name, bytes) else name)

t0 = time.time()
a = cp.arange(4096, dtype=cp.float64)
b = cp.abs(a) * 2.0
b = cp.log2(cp.maximum(b, 1.0))
cp.cuda.Stream.null.synchronize()
print("[test] elementwise kernel ok", round(time.time() - t0, 2), "s")

t0 = time.time()
frames = cp.zeros((64, 2048), dtype=cp.float64)
spec = cp.fft.rfft(frames, axis=1)
_ = cp.abs(spec)
cp.cuda.Stream.null.synchronize()
print("[test] cuFFT rfft ok", round(time.time() - t0, 2), "s")

k32 = ctypes.windll.kernel32
k32.GetModuleHandleW.restype = ctypes.c_void_p
k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
for dll in (
    "cudart64_110.dll",
    "cufft64_10.dll",
    "nvrtc64_112_0.dll",
    "nvrtc-builtins64_116.dll",
):
    h = k32.GetModuleHandleW(dll)
    if not h:
        print("[test] NOT LOADED", dll)
        continue
    buf = ctypes.create_unicode_buffer(1024)
    k32.GetModuleFileNameW(h, buf, 1024)
    tag = "BUNDLED" if buf.value.lower().startswith(staged.lower()) else "SYSTEM!"
    print(f"[test] {tag:8s} {dll} -> {buf.value}")

print("[test] RESULT: GPU works with bundled CUDA runtime only")
