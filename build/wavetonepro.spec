# -*- mode: python ; coding: utf-8 -*-
r"""WavetonePro 打包配置（PyInstaller onedir）。

本文件住在 <项目>\build\ 里；入口脚本是上一级的 wavetonepro.py。
详细踩坑记录见项目根目录的「编译日志.md」，用法见「打包说明.md」。

为什么这样配：
  * 只依赖一个干净的 venv（build\venv，由 build_exe.ps1 创建），torch / gradio /
    cv2 / pandas 这些无关库根本不在环境里，也就不会被扫进来 ——
    之前 3GB 的体积主要来自它们。
  * CUDA 运行库只打包真正用到的 4 个 DLL（cudart / cufft / nvrtc /
    nvrtc-builtins，合计约 384MB）。cublas / cublasLt / cusolver / cusparse /
    curand / cudnn 合计约 970MB 全部剔除：wavetonepro.py 里没有任何
    dot / matmul / einsum / linalg / sparse / random 调用，实测运行时只加载
    cufft + nvrtc + nvtx，这些库永远不会被 import。
  * cufft / nvrtc 是 CuPy **按文件名** 动态加载的（不是静态导入），
    PyInstaller 的依赖分析看不见它们，必须显式放进 cuda/bin，再由运行时钩子
    build/rthook_cuda.py 加进 DLL 搜索路径。
  * CuPy 的 JIT 需要 cupy/_core/include 下的头文件，属于 data，必须手动收集。

开关可以用环境变量覆盖（build_exe.ps1 就是这么驱动的）：
    WAVETONEPRO_BUNDLE_CUDA=0      -> 纯 CPU 版
    WAVETONEPRO_INCLUDE_SCIPY=1    -> 带上 scipy（+130MB）
    WAVETONEPRO_CONSOLE=0          -> 不显示控制台
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


def _env_flag(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


# --------------------------------------------------------------------------
# 开关
# --------------------------------------------------------------------------
BUNDLE_CUDA = _env_flag("WAVETONEPRO_BUNDLE_CUDA", True)
INCLUDE_SCIPY = _env_flag("WAVETONEPRO_INCLUDE_SCIPY", False)
SHOW_CONSOLE = _env_flag("WAVETONEPRO_CONSOLE", True)

APP_NAME = "WavetonePro" if BUNDLE_CUDA else "WavetonePro-CPU"

# CuPy 需要的 CUDA 运行库（11.x 命名，对应 cupy-cuda11x）
CUDA_DLLS = [
    "cudart64_110.dll",
    "cufft64_10.dll",
    "nvrtc64_112_0.dll",
    "nvrtc-builtins64_116.dll",
]

# 这些 CUDA Toolkit 的 DLL 一律不要（PyInstaller 可能顺着 cupy 的 .pyd
# 静态导入把它们从 PATH 上捞进来，那就是白送的几百 MB）
DROP_DLL_PREFIXES = (
    "cublas",
    "cusolver",
    "cusparse",
    "curand",
    "cudnn",
    "cutensor",
    "npp",
    "nvjpeg",
    "nvblas",
    "cuinj",
    "cufftw",
    "cudart32",
    "cufft32",
)

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
# 本 spec 住在 <项目>\build\ 里：SPECPATH 就是 build 目录，
# 项目根（wavetonepro.py 所在）是它的上一级。
BUILD_DIR = os.path.abspath(SPECPATH)
PROJECT_DIR = os.path.dirname(BUILD_DIR)
CUDA_STAGE_DIR = os.path.join(BUILD_DIR, "cuda")
ENTRY_SCRIPT = os.path.join(PROJECT_DIR, "wavetonepro.py")

# --------------------------------------------------------------------------
# 显式依赖（动态 import / C 层 import，静态分析看不到的）
# --------------------------------------------------------------------------
# 坑点：cupy 内部大量使用 Cython 的 `cimport`，Cython 会在模块初始化时发出一条
# **纯 C 层**的 import —— PyInstaller 的字节码分析完全看不见。实测漏掉
# cupy_backends.cuda._softlink 会让 import cupy 直接 ModuleNotFoundError，
# 漏掉 cupy._core._carray 也一样。所以这里把 cupy / cupy_backends 的子模块
# **整包收全**，只用 excludes 挡掉不想要的大件（见下面的 CUDA 相关排除项）。
def _keep_cupy_submodule(name):
    # cupy.testing 会拖进 pytest，纯属浪费；thrust 是可选的，本程序用不到
    if ".testing" in name:
        return False
    if name == "cupy.cuda.thrust":
        return False
    return True


hiddenimports = collect_submodules("cupy", filter=_keep_cupy_submodule)
hiddenimports += collect_submodules("cupy_backends")
hiddenimports += [
    "soundfile",
    "mido",
    "mido.backends.rtmidi",  # mido 通过 importlib 动态载入后端
]
if BUNDLE_CUDA:
    # cupy 用 Cython 的 cimport 引 fastrlock，同样是静态分析看不见的 C 层导入
    hiddenimports += collect_submodules("fastrlock")

if not BUNDLE_CUDA:
    hiddenimports = [
        m for m in hiddenimports if not m.startswith("cupy") and not m.startswith("cuda")
    ]

# --------------------------------------------------------------------------
# 数据文件：CuPy 的 JIT 头文件（缺了它 GPU 内核编译会失败）
# --------------------------------------------------------------------------
datas = []
if BUNDLE_CUDA:
    datas += collect_data_files("cupy", includes=["_core/include/**", ".data/*.json"])

# --------------------------------------------------------------------------
# 二进制
# --------------------------------------------------------------------------
binaries = []

if BUNDLE_CUDA:
    # 1) 自带的 CUDA 运行库（soundfile 的 libsndfile 由 hooks-contrib 处理）
    for _name in CUDA_DLLS:
        _src = os.path.join(CUDA_STAGE_DIR, "bin", _name)
        if not os.path.isfile(_src):
            raise SystemExit(
                "缺少 CUDA 运行库: %s\n"
                "先执行:  powershell -ExecutionPolicy Bypass -File build\\build_exe.ps1\n"
                "（脚本会从系统 CUDA Toolkit 复制到 build\\cuda\\bin）" % _src
            )
        binaries.append((_src, os.path.join("cuda", "bin")))

    # 2) MSVC 运行时：cupy 的 .pyd 静态依赖 MSVCP140.dll，而它住在 System32，
    #    PyInstaller 把它当“系统 DLL”直接跳过，干净系统上就会 import 失败。
    _sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for _name in ("msvcp140.dll", "concrt140.dll"):
        _src = os.path.join(_sys32, _name)
        if os.path.isfile(_src):
            binaries.append((_src, "."))

# --------------------------------------------------------------------------
# 排除项：这些库可能装在这台机器上，但和本项目毫无关系
# --------------------------------------------------------------------------
excludes = [
    # 深度学习 / 科学计算全家桶
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "keras",
    "jax",
    "onnx",
    "onnxruntime",
    "transformers",
    "sklearn",
    "sympy",
    "numba",
    "llvmlite",
    "pandas",
    "matplotlib",
    "mpl_toolkits",
    "PIL",
    "cv2",
    "plotly",
    "gradio",
    "playwright",
    "selenium",
    "yt_dlp",
    "imageio",
    "pyarrow",
    "numpy.f2py",
    "numpy.distutils",
    # 其它界面 / 交互环境
    "tkinter",
    "_tkinter",
    "IPython",
    "jupyter",
    "notebook",
    "nbformat",
    "ipykernel",
    "PySide2",
    "PySide6",
    "PyQt6",
    # 打包工具自身
    "pytest",
    "pydoc_data",
    # 不需要的 CUDA 绑定（配套的 DLL 也被 DROP_DLL_PREFIXES 挡掉）
    "cupy_backends.cuda.libs.cublas",
    "cupy_backends.cuda.libs.cusolver",
    "cupy_backends.cuda.libs.cusparse",
    "cupy_backends.cuda.libs.curand",
    "cupy_backends.cuda.libs.cudnn",
    "cupy_backends.cuda.libs.cutensor",
    # Thrust 是可选加速后端（cupy/cuda/__init__.py 里用 try/except ImportError
    # 包着），wavetonepro.py 从不用 sort/scan/cumsum，砍掉省 45MB。
    # 注意 cub 不能砍：cupy/cuda/__init__.py 第 34 行是无条件导入。
    "cupy.cuda.thrust",
]

if not INCLUDE_SCIPY:
    excludes.append("scipy")

if not BUNDLE_CUDA:
    # 纯 CPU 版。wavetonepro.py 里的 `import cupy as _cp` 都写在 try/except 里，
    # 但 PyInstaller 的静态分析照样会收 —— 必须在这里整体排除，否则 cupy 的
    # 108MB + 它静态依赖的 cufft64_10.dll（345MB）会一起被打进来。
    excludes += ["cupy", "cupy_backends", "cupyx", "fastrlock", "cuda"]

# --------------------------------------------------------------------------
# 分析
# --------------------------------------------------------------------------
a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[PROJECT_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(BUILD_DIR, "rthook_cuda.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)


# 兜底过滤。注意 a.binaries 是 PyInstaller 的 TOC，元组顺序是
# (目标相对路径, 源绝对路径, 类型)，不是 (源, 目标)。
def _is_dropped_cuda_dll(entry):
    base = os.path.basename(entry[0]).lower()
    if not base.endswith(".dll"):
        return False
    if base in {n.lower() for n in CUDA_DLLS}:
        return False  # 显式打包的 4 个，留着
    return base.startswith(DROP_DLL_PREFIXES)


def _toc_size_mb(entries):
    return sum(
        os.path.getsize(e[1]) for e in entries if os.path.isfile(e[1])
    ) / (1024 * 1024)


_dropped = [b for b in a.binaries if _is_dropped_cuda_dll(b)]
if _dropped:
    a.binaries = [b for b in a.binaries if not _is_dropped_cuda_dll(b)]
    print(
        "[spec] drop %d unwanted CUDA DLLs (~%.0f MB): %s"
        % (len(_dropped), _toc_size_mb(_dropped), ", ".join(sorted({os.path.basename(b[0]) for b in _dropped})))
    )


# cufft64_10.dll / nvrtc64_112_0.dll 同时是 cupy 某些 .pyd 的**静态导入**，
# PyInstaller 会顺着依赖把它们也塞进 _internal 根目录，于是和 cuda/bin 里的
# 那份重复，白白多出 376MB。这里只保留 cuda/bin 里的那一份 —— 运行时钩子
# 已经把 cuda/bin 加进了 DLL 搜索路径，静态导入同样能解析到。
_CUDA_BIN_DEST = os.path.normpath(os.path.join("cuda", "bin"))
_cuda_names = {n.lower() for n in CUDA_DLLS}


def _is_redundant_cuda_copy(entry):
    if os.path.basename(entry[0]).lower() not in _cuda_names:
        return False
    # entry[0] 是目标**文件**路径，取它所在目录跟 cuda\bin 比
    return os.path.normpath(os.path.dirname(entry[0])) != _CUDA_BIN_DEST


_redundant = [b for b in a.binaries if _is_redundant_cuda_copy(b)]
if _redundant:
    a.binaries = [b for b in a.binaries if not _is_redundant_cuda_copy(b)]
    print(
        "[spec] drop %d duplicate CUDA DLL copies (~%.0f MB): %s"
        % (len(_redundant), _toc_size_mb(_redundant), ", ".join(sorted({os.path.basename(b[0]) for b in _redundant})))
    )

# 打包后自检用：确认 cuda/bin 里 4 个 DLL 都在
_kept = sorted(os.path.normpath(b[0]) for b in a.binaries if os.path.basename(b[0]).lower() in _cuda_names)
print("[spec] bundled CUDA DLLs: %s" % (", ".join(_kept) or "(none)"))

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX 会和 CUDA / Qt 的 DLL 打架，别开
    console=SHOW_CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
