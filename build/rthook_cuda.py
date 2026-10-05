# -*- coding: utf-8 -*-
"""PyInstaller 运行时钩子 —— 让打包后的 WavetonePro 找到自带的 CUDA 运行库。

在用户代码之前执行。三件事：

1. 把 `<程序目录>/cuda/bin` 加进 Windows 的 DLL 搜索路径，并把 `CUDA_PATH`
   指向 `<程序目录>/cuda`。
   - cupy 的 `_setup_win32_dll_directory()` 会自己执行
     `os.add_dll_directory(CUDA_PATH + "/bin")`，所以 CUDA_PATH 必须是一个
     真的含有 bin 子目录的路径，否则 import cupy 时直接抛异常。
   - cudart64_110.dll / cufft64_10.dll / nvrtc64_112_0.dll 都是 cupy
     **按文件名** 动态加载的（不是静态导入），所以只能靠搜索路径找到。

2. 自己再做一遍 PATH + add_dll_directory 兜底。
   ⚠ `os.add_dll_directory()` 返回的句柄一旦被 GC，该目录就会从搜索路径里
   消失，所以句柄必须保存在模块级列表里。

3. 把 CuPy 的 JIT 内核缓存指到可写目录。
   默认缓存目录（~/.cupy/kernel_cache）不可写时，CuPy 编译内核会**永久卡死**
   而不是报错（空转一个 CPU 核），这是实测到的真实故障模式。

附带 `--cuda-selftest` 诊断模式：不启动界面，只打印 CUDA 可用性与实际
加载到的 DLL 路径，用于在没有 Python 的机器上排查 GPU 问题。
"""

import os
import sys

# 必须持有句柄引用，否则目录会被移出 DLL 搜索路径
_DLL_HANDLES = []


def _bundle_dir():
    """onedir 打包下是 exe 旁边的 _internal；onefile 下是解压出来的临时目录。"""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return base
    return os.path.dirname(os.path.abspath(sys.executable))


def _setup_cuda():
    base = _bundle_dir()
    cuda_bin = os.path.join(base, "cuda", "bin")
    if not os.path.isdir(cuda_bin):
        # 纯 CPU 版（没有打包 CUDA 运行库）：什么都不做，
        # 也绝不能设置 CUDA_PATH，否则 cupy 会去找不存在的 bin 目录而报错。
        return
    cuda_root = os.path.dirname(cuda_bin)

    # 1) CUDA_PATH：cupy 自己会用它做 add_dll_directory
    os.environ["CUDA_PATH"] = cuda_root

    # 2) PATH 前置 + 原生 DLL 目录（两者都做，覆盖不同版本的加载方式）
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    for d in (cuda_bin, base):
        try:
            _DLL_HANDLES.append(os.add_dll_directory(d))
        except OSError:
            pass


def _user_data_dir():
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(root, "WavetonePro")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    return d


def _setup_cupy_cache():
    """CuPy JIT 内核缓存放到确定可写的位置。

    默认缓存目录（~/.cupy/kernel_cache）不可写时，CuPy 编译内核不会报错，
    而是**永久空转**（实测：占满一个 CPU 核、界面卡死），所以必须显式指定。
    """
    if os.environ.get("CUPY_CACHE_DIR"):
        return
    d = _user_data_dir()
    if d:
        os.environ["CUPY_CACHE_DIR"] = os.path.join(d, "cupy_cache")


def _setup_gpu_cache():
    """把 GPU 探测缓存放进用户目录，而不是打包目录里。

    wavetonepro.py 的 _gpu_cache_path() 会优先用 WAVETONEPRO_GPU_CACHE。
    这一点很重要：缓存只记录“成功”，如果缓存文件留在程序目录里跟着文件夹
    一起拷到一台没有 N 卡的机器，程序会误判 GPU 可用、直接进 GPU 模式后崩掉。
    放到 %LOCALAPPDATA% 就天然变成“每台机器各自探测”。
    """
    if os.environ.get("WAVETONEPRO_GPU_CACHE"):
        return
    d = _user_data_dir()
    if d:
        os.environ["WAVETONEPRO_GPU_CACHE"] = os.path.join(d, ".wavetonepro_gpu.json")


def _check_gpu():
    """返回 (是否致命失败, 说明)。GPU 存在但算不了 = 致命；没有 N 卡 = 不致命。"""
    import ctypes

    try:
        import cupy as cp
    except Exception as e:
        if not os.path.isdir(os.path.join(_bundle_dir(), "cuda", "bin")):
            # 纯 CPU 版本来就不带 CuPy，属于正常
            print("  [跳过] 这是纯 CPU 版（没打包 CuPy），用 NumPy 计算")
            return True, ""
        print(f"  [失败] 无法 import cupy: {type(e).__name__}: {e}")
        return False, "这台机器只能用 CPU，这个报错说明打包不完整"

    print("  CuPy 版本    :", cp.__version__)

    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        print(f"  [失败] 无法访问 CUDA 驱动: {type(e).__name__}: {e}")
        return False, "请确认已安装 NVIDIA 显卡驱动"

    if n <= 0:
        print("  [跳过] 未检测到 NVIDIA 显卡 -> 用 CPU 计算（正常）")
        return True, ""

    try:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        name = name.decode("utf-8", "ignore") if isinstance(name, bytes) else str(name)
    except Exception:
        name = "CUDA"
    print("  显卡         :", name)

    try:
        # 真正跑一次 kernel：会触发 NVRTC 编译，需要 nvrtc-builtins
        a = cp.zeros(4, dtype=cp.float64)
        a += 1.0
        cp.cuda.Stream.null.synchronize()
        assert float(cp.asnumpy(a)[0]) == 1.0
        print("  基础 kernel  : OK")

        # cuFFT：分析流程里的 xp.fft.rfft 走这条路径
        frames = cp.zeros((8, 1024), dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        spec = cp.fft.rfft(frames, axis=1)
        cp.cuda.Stream.null.synchronize()
        print("  cuFFT rfft   : OK  ->", spec.shape)
    except Exception as e:
        print(f"  [失败] GPU 计算失败: {type(e).__name__}: {e}")
        return False, "GPU 存在但算不了，检查显卡驱动是否正常"

    print("  实际加载的 CUDA 动态库：")
    k32 = ctypes.windll.kernel32
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    base = _bundle_dir().lower()
    for dll in (
        "nvcuda.dll",
        "cudart64_110.dll",
        "cufft64_10.dll",
        "nvrtc64_112_0.dll",
        "nvrtc-builtins64_116.dll",
    ):
        handle = k32.GetModuleHandleW(dll)
        if not handle:
            print(f"    (未加载)  {dll}")
            continue
        buf = ctypes.create_unicode_buffer(1024)
        k32.GetModuleFileNameW(handle, buf, 1024)
        where = "自带" if buf.value.lower().startswith(base) else "系统"
        print(f"    [{where}]  {dll}")
    return True, ""


def _check_audio():
    try:
        import soundfile as sf

        print("  libsndfile   :", sf.__libsndfile_version__)
    except Exception as e:
        print(f"  [失败] soundfile 不可用: {type(e).__name__}: {e}")
        return False, "音频解码会退回到只支持 PCM WAV 的内置解码器"

    # 有给文件就真解一次
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    if paths and os.path.isfile(paths[0]):
        try:
            data, sr = sf.read(paths[0], dtype="float32", always_2d=True)
            print(f"  解码测试     : OK  {os.path.basename(paths[0])}  "
                  f"{data.shape[0]} 帧 / {sr} Hz / {data.shape[1]} 声道")
        except Exception as e:
            print(f"  [失败] 解码 {paths[0]} 失败: {type(e).__name__}: {e}")
            return False, ""
    return True, ""


def _check_qt():
    try:
        from PyQt5.QtCore import QT_VERSION_STR
        from PyQt5.QtWidgets import QApplication  # noqa: F401

        print("  Qt 版本      :", QT_VERSION_STR)
    except Exception as e:
        print(f"  [失败] PyQt5 不可用: {type(e).__name__}: {e}")
        return False, "界面起不来"

    try:
        from PyQt5.QtMultimedia import QMediaPlayer  # noqa: F401

        print("  QtMultimedia : OK（播放功能可用）")
    except Exception as e:
        print(f"  [警告] QtMultimedia 不可用: {type(e).__name__}: {e}")
        print("         界面仍可用，但没有试听播放")
    return True, ""


def _check_midi():
    try:
        import mido
    except Exception as e:
        print(f"  [失败] mido 不可用: {type(e).__name__}: {e}")
        return True, "点频谱不会发声"

    try:
        outputs = mido.get_output_names()
    except Exception as e:
        print(f"  [警告] 无法枚举 MIDI 输出: {type(e).__name__}: {e}")
        return True, "点频谱不会发声"

    if outputs:
        print("  MIDI 输出    :", ", ".join(outputs))
    else:
        print("  [警告] 没有可用的 MIDI 输出（点频谱不会发声）")
    return True, ""


def _selftest():
    """`WavetonePro.exe --selftest`：只做环境自检，不开界面。"""
    print("=" * 68)
    print("WavetonePro 环境自检")
    print("=" * 68)
    print("程序目录      :", _bundle_dir())
    print("CUDA_PATH     :", os.environ.get("CUDA_PATH", "(未设置，纯 CPU 版)"))
    print("CUPY_CACHE_DIR:", os.environ.get("CUPY_CACHE_DIR", "(默认)"))
    print("GPU 探测缓存  :", os.environ.get("WAVETONEPRO_GPU_CACHE", "(默认)"))

    results = []
    for title, fn in (
        ("[1/4] GPU 计算 (CuPy / CUDA)", _check_gpu),
        ("[2/4] 音频解码 (soundfile)", _check_audio),
        ("[3/4] 界面 (PyQt5)", _check_qt),
        ("[4/4] MIDI 输出", _check_midi),
    ):
        print("-" * 68)
        print(title)
        try:
            ok, hint = fn()
        except Exception as e:
            import traceback

            traceback.print_exc()
            ok, hint = False, f"{type(e).__name__}: {e}"
        results.append((title, ok, hint))

    print("=" * 68)
    failed = [r for r in results if not r[1]]
    for title, ok, hint in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {title}" + (f"    -> {hint}" if hint else ""))
    print("=" * 68)
    if failed:
        print("结论：有项目没通过，见上面各节的 [失败] 说明。")
        return 1
    print("结论：环境正常，可以正常使用。")
    return 0


def _main():
    _setup_cuda()
    _setup_cupy_cache()
    _setup_gpu_cache()

    if "--selftest" in sys.argv or "--cuda-selftest" in sys.argv:
        try:
            code = _selftest()
        except Exception:  # 自检本身出错也要给出可读信息
            import traceback

            traceback.print_exc()
            code = 1
        sys.stdout.flush()
        sys.exit(code)


_main()
