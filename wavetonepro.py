import sys
import os
import io
import json
import math
import wave
import tempfile
import threading

import numpy as np

# =========================================================================
# 后端探测
# -------------------------------------------------------------------------
# 探测在本进程的后台线程里做，绝不在 import 期执行 CUDA 调用。
# 某些环境下 CUDA 上下文能建、显存能分配，但第一次 kernel launch 会永久
# 阻塞；若放在模块顶层，程序连窗口都到不了。详见 _probe_cuda / _start_gpu_probe。
# =========================================================================
HAS_GPU = False  # 当前是否真正启用 GPU 计算
CUPY_AVAILABLE = False  # 是否安装了 cupy（仅表示可尝试）
GPU_NAME = "CPU (NumPy)"
GPU_STATUS = "CPU (NumPy)"  # 给状态栏用的描述
XP_LOCK = threading.Lock()  # xp 只允许在两次分析之间切换
xp = np

_GPU_PROBE_DONE = None  # threading.Event：探测是否已结束
_GPU_PROBE_WAIT = 5.0  # 分析前最多等探测这么久（秒）
_PROBE_START_LOCK = threading.Lock()  # 保证探测只启动一次

# 环境变量开关（可选）：
#   WAVETONEPRO_NO_GPU=1    强制 CPU
#   WAVETONEPRO_FORCE_GPU=1 强制尝试 GPU（跳过自检）
#   WAVETONEPRO_GPU_WAIT=0  分析前不等探测（探测仍会后台跑，下次分析生效）
_ENV_NO_GPU = os.environ.get("WAVETONEPRO_NO_GPU", "") not in ("", "0")
_ENV_FORCE_GPU = os.environ.get("WAVETONEPRO_FORCE_GPU", "") not in ("", "0")
try:
    _GPU_PROBE_WAIT = max(0.0, float(os.environ.get("WAVETONEPRO_GPU_WAIT", _GPU_PROBE_WAIT)))
except Exception:
    pass


def _gpu_cache_path():
    """选一个真正可写的位置存放探测结果。

    不能只认 ~ —— 受限环境（沙箱/只读用户目录）下写入会失败。
    按 环境变量 -> 脚本目录 -> 用户目录 -> 临时目录 依次尝试。
    """
    env = os.environ.get("WAVETONEPRO_GPU_CACHE")
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    candidates = [
        p
        for p in (
            env,
            os.path.join(base, ".wavetonepro_gpu.json"),
            os.path.join(os.path.expanduser("~"), ".wavetonepro_gpu.json"),
            os.path.join(tempfile.gettempdir(), "wavetonepro_gpu.json"),
        )
        if p
    ]
    for path in candidates:
        try:
            d = os.path.dirname(path)
            if d and not os.path.isdir(d):
                continue
            with open(path, "a", encoding="utf-8"):
                pass
            return path
        except Exception:
            continue
    return candidates[-1]


def _load_gpu_cache():
    """读取缓存。

    只返回**成功**结果。失败的结论绝不落盘也不复用 —— 同一台机器换个
    启动方式（普通终端 / 受限沙箱）CUDA 可用性可能完全不同，把一次失败
    记下来会导致"以前能用 GPU，现在只能用 CPU"这种倒退。
    """
    try:
        with open(_gpu_cache_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("ok"):
            return True, str(d.get("name", ""))
    except Exception:
        pass
    return None


def _save_gpu_cache(ok, name):
    """只缓存成功。失败结果不写盘。"""
    if not ok:
        return
    try:
        with open(_gpu_cache_path(), "w", encoding="utf-8") as f:
            json.dump({"ok": True, "name": str(name)}, f)
    except Exception:
        pass


def _probe_cuda():
    """**在应用自己的进程里**测一次真实 kernel launch。

    必须在本进程做：探测环境要和实际计算环境一致。放进子进程的话，
    子进程的环境与宿主启动方式（终端/沙箱/服务）可能不同，会得出
    与实际相反的结论。

    返回 (ok, device_name)。异常一律视为不可用。
    """
    try:
        import cupy as _cp
    except Exception as e:
        print(f"[gpu] 无法 import cupy: {e}")
        return False, ""
    try:
        if _cp.cuda.runtime.getDeviceCount() <= 0:
            print("[gpu] 未检测到 CUDA 设备 -> 使用 CPU")
            return False, ""
        a = _cp.zeros(1)
        a += 1
        _cp.cuda.Stream.null.synchronize()
        if float(_cp.asnumpy(a)[0]) != 1.0:
            print("[gpu] 自检数值异常 -> 使用 CPU")
            return False, ""
        try:
            p = _cp.cuda.runtime.getDeviceProperties(0)["name"]
            name = p.decode("utf-8", "ignore") if isinstance(p, bytes) else str(p)
        except Exception:
            name = "CUDA"
        return True, name
    except Exception as e:
        print(f"[gpu] 自检失败: {type(e).__name__}: {e} -> 使用 CPU")
        return False, ""


def _activate_backend(use_gpu, name=""):
    """切换计算后端。只在 AnalysisWorker 没在跑的时候调用。"""
    global HAS_GPU, GPU_NAME, GPU_STATUS, xp
    try:
        import cupy as _cp
    except Exception:
        use_gpu = False
    if use_gpu:
        with XP_LOCK:
            xp = _cp
            HAS_GPU = True
            GPU_NAME = f"GPU ({name})" if name else "GPU (CUDA)"
            GPU_STATUS = GPU_NAME
        print(f"[gpu] 已启用 {GPU_NAME}")
    else:
        with XP_LOCK:
            xp = np
            HAS_GPU = False
            GPU_NAME = "CPU (NumPy)"
            GPU_STATUS = "CPU (NumPy)"


def _ensure_backend_ready():
    """在开始分析前确保后端已经定下来。

    探测在本进程的后台线程里跑。若它已经在跑（正常情况，启动时就跑了），
    这里只等一小会儿；没等到就用 CPU 继续 —— 绝不让分析卡住。
    """
    if _GPU_PROBE_DONE is None:
        _start_gpu_probe()
    if _GPU_PROBE_DONE is not None:
        _GPU_PROBE_DONE.wait(_GPU_PROBE_WAIT)


def _gpu_probe_thread():
    try:
        if _ENV_NO_GPU:
            print("[gpu] WAVETONEPRO_NO_GPU 已设置 -> 强制 CPU")
            _activate_backend(False)
            return
        if _ENV_FORCE_GPU:
            import cupy as _cp

            try:
                p = _cp.cuda.runtime.getDeviceProperties(0)["name"]
                name = p.decode("utf-8", "ignore") if isinstance(p, bytes) else str(p)
            except Exception:
                name = "CUDA"
            print("[gpu] WAVETONEPRO_FORCE_GPU 已设置 -> 跳过自检直接启用")
            _activate_backend(True, name)
            return

        cached = _load_gpu_cache()
        if cached is not None:
            # 只缓存过成功结果：直接启用，省掉一次自检
            ok, name = cached
        else:
            ok, name = _probe_cuda()
            _save_gpu_cache(ok, name)
        _activate_backend(ok, name)
    except Exception as e:
        print(f"[gpu] 探测线程异常: {type(e).__name__}: {e}")
        _activate_backend(False)
    finally:
        # 先通知界面（此时还没 set 事件，主线程若在等会被正常唤醒）
        _notify_backend_changed(GPU_STATUS)
        if _GPU_PROBE_DONE is not None:
            _GPU_PROBE_DONE.set()


def _start_gpu_probe():
    """非阻塞启动后端探测；重复调用只会生效一次。

    _PROBE_START_LOCK 防止"检查-赋值"竞态：启动时的 main() 与首次分析前的
    _ensure_backend_ready() 可能分别在不同线程里调用到这里，若同时通过
    `_GPU_PROBE_DONE is None` 的判断，就会起两个探测线程、且后一个把
    前一个的 Event 覆盖掉，导致等待方永远等不到。
    """
    global CUPY_AVAILABLE, _GPU_PROBE_DONE
    with _PROBE_START_LOCK:
        if _GPU_PROBE_DONE is not None:
            return  # 已经探测过/正在探测
        try:
            import importlib.util

            CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
        except Exception:
            CUPY_AVAILABLE = False

        done = threading.Event()
        _GPU_PROBE_DONE = done
        if not CUPY_AVAILABLE and not _ENV_FORCE_GPU:
            _activate_backend(False)
            done.set()
            return
    # 线程在锁外启动，避免探测线程过早回来抢同一把锁
    threading.Thread(target=_gpu_probe_thread, name="gpu-probe", daemon=True).start()


# =========================================================================
# MIDI 输出
# =========================================================================
MIDI_AVAILABLE = False
MIDI_BACKEND = None
MIDI_PORT_NAME = ""
_midi_out = None


def _init_midi():
    global MIDI_AVAILABLE, MIDI_BACKEND, MIDI_PORT_NAME, _midi_out
    _midi_out = None
    MIDI_PORT_NAME = ""

    try:
        import mido

        try:
            ports = mido.get_output_names()
        except Exception:
            ports = []
        if ports:
            target = None
            for kw in ("GS Wavetable", "Microsoft GS", "Wavetable", "Synth"):
                for p in ports:
                    if kw in p:
                        target = p
                        break
                if target:
                    break
            if target is None:
                target = ports[0]
            _midi_out = mido.open_output(target)
            MIDI_BACKEND = "mido"
            MIDI_PORT_NAME = target
            MIDI_AVAILABLE = True
            return
    except Exception as e:
        print(f"[midi] mido 初始化失败: {e}")

    try:
        import pygame.midi as pmidi

        pmidi.init()
        n = pmidi.get_count()
        candidates = []
        for i in range(n):
            info = pmidi.get_device_info(i)
            if info[3]:
                name = info[1]
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "ignore")
                candidates.append((i, name))
        if not candidates:
            return
        target = None
        for i, name in candidates:
            if "GS" in name or "Wavetable" in name or "Synth" in name:
                target = (i, name)
                break
        if target is None:
            target = candidates[0]
        i, name = target
        _midi_out = pmidi.Output(i)
        MIDI_BACKEND = "pygame"
        MIDI_PORT_NAME = name
        MIDI_AVAILABLE = True
    except Exception as e:
        print(f"[midi] pygame.midi 初始化失败: {e}")


def midi_note_on(note, vel=90):
    if not MIDI_AVAILABLE or _midi_out is None:
        return
    try:
        if MIDI_BACKEND == "mido":
            import mido

            _midi_out.send(mido.Message("note_on", note=int(note), velocity=int(vel)))
        else:
            _midi_out.note_on(int(note), int(vel))
    except Exception as e:
        print(f"[midi] note_on 失败: {e}")


def midi_note_off(note):
    if not MIDI_AVAILABLE or _midi_out is None:
        return
    try:
        if MIDI_BACKEND == "mido":
            import mido

            _midi_out.send(mido.Message("note_off", note=int(note), velocity=0))
        else:
            _midi_out.note_off(int(note), 0)
    except Exception as e:
        print(f"[midi] note_off 失败: {e}")


# =========================================================================
# PyQt
# =========================================================================
from PyQt5.QtCore import (
    Qt,
    QObject,
    QRectF,
    QPointF,
    QSize,
    pyqtSignal,
    QTimer,
    QUrl,
    QThread,
)
from PyQt5.QtGui import QImage, QPainter, QColor, QPen, QFont, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QComboBox,
    QSizePolicy,
    QMessageBox,
    QToolBar,
    QAction,
    QSlider,
    QStatusBar,
    QProgressBar,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QGroupBox,
    QCheckBox,
)

try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

    HAS_MEDIA = True
except Exception as _e:
    QMediaPlayer = None
    QMediaContent = None
    HAS_MEDIA = False
    print(f"[media] QtMultimedia 不可用: {_e}")


# =========================================================================
# 后端切换通知（探测线程 → 主线程刷新界面）
# -------------------------------------------------------------------------
# 界面是在探测开始之前就建好的，标题栏和状态栏里存的是那一刻的快照。
# 探测在后台完成后必须把结果推回主线程，否则界面会一直显示 "CPU (NumPy)"，
# 即使实际已经在用 GPU 计算。控件只能在主线程改，所以走信号。
# =========================================================================
class _BackendNotifier(QObject):
    changed = pyqtSignal(str)


_notifier = _BackendNotifier()


def _notify_backend_changed(name):
    """线程安全：从探测线程发信号给主线程。"""
    try:
        _notifier.changed.emit(str(name))
    except Exception as e:
        print(f"[gpu] 界面通知失败: {e}")


def _backend_label():
    """状态栏用的后端描述（在探测结束后会变，所以要能重建）。"""
    if HAS_GPU and GPU_NAME.startswith("GPU ("):
        inner = GPU_NAME[5:-1]
        if len(inner) <= 28:
            return GPU_NAME
        return f"GPU ({inner[:25]}…)"
    return "CPU (NumPy)"


def _backend_short():
    return "GPU" if HAS_GPU else "CPU"


def _midi_label():
    return f"MIDI: {MIDI_PORT_NAME}" if MIDI_AVAILABLE else "MIDI: 不可用"


# =========================================================================
# 常量
# =========================================================================
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BLACK_PC = {1, 3, 6, 8, 10}

MIDI_MIN = 21
MIDI_MAX = 108
# 显示音域滑块的可调范围。两边跨度都锁成 24 个半音，且
# KEY_SPAN_LO_MAX + (MIDI_MAX - KEY_SPAN_HI_MIN) == MIDI_MAX - MIDI_MIN，
# 所以两个滑块都拉到头正好是完整键盘，中间不会出现够不着的空档。
KEY_SPAN_LO_MAX = MIDI_MIN + 24      # 最低音最多调到 45 (A2)
KEY_SPAN_HI_MIN = MIDI_MAX - 24      # 最高音最少调到 84 (C6)
MIN_KEY_SPAN = 2                     # 显示音域至少保留 2 个半音
MASK_MIN_DRAG_PX = 3                 # 右键至少拖动这么多像素才创建遮罩
DEFAULT_ROWS_PER_SEMITONE = 12
ROWS_PER_SEMITONE_CHOICES = (1, 6, 12, 24, 48)

# -------------------------------------------------------------------------
# 和弦模式
# -------------------------------------------------------------------------
# 每个和弦用"从**低音**往上数的半音间隔"描述。所以转位不改变低音 ——
# 鼠标指在哪个音上，哪个音就是低音，这点和"同时画几个音符"的直觉一致。
# 例如 C 上的大三和弦第二转位（第一转位）: [0, 5, 9] -> C F A。
CHORDS = (
    # ---- 三和弦 ----
    ("大三和弦 (1-3-5)", 2, (0, 4, 7)),
    ("小三和弦 (1-b3-5)", 2, (0, 3, 7)),
    ("减三和弦 (1-b3-b5)", 2, (0, 3, 6)),
    ("增三和弦 (1-3-#5)", 2, (0, 4, 8)),
    ("挂四和弦 (1-4-5)", 2, (0, 5, 7)),
    ("挂二和弦 (1-2-5)", 2, (0, 2, 7)),
    # ---- 七和弦 ----
    ("属七和弦 (1-3-5-b7)", 3, (0, 4, 7, 10)),
    ("大七和弦 (1-3-5-7)", 3, (0, 4, 7, 11)),
    ("小七和弦 (1-b3-5-b7)", 3, (0, 3, 7, 10)),
    ("小大七和弦 (1-b3-5-7)", 3, (0, 3, 7, 11)),
    ("半减七和弦 (1-b3-b5-b7)", 3, (0, 3, 6, 10)),
    ("减七和弦 (1-b3-b5-bb7)", 3, (0, 3, 6, 9)),
    ("增七和弦 (1-3-#5-b7)", 3, (0, 4, 8, 10)),
    ("大六和弦 (1-3-5-6)", 3, (0, 4, 7, 9)),
    ("小六和弦 (1-b3-5-6)", 3, (0, 3, 7, 9)),
    # ---- 九和弦 / 延伸 ----
    ("大九和弦 (1-3-5-7-9)", 4, (0, 4, 7, 11, 14)),
    ("属九和弦 (1-3-5-b7-9)", 4, (0, 4, 7, 10, 14)),
    ("小九和弦 (1-b3-5-b7-9)", 4, (0, 3, 7, 10, 14)),
    ("加九和弦 (1-3-5-9)", 3, (0, 4, 7, 14)),
    ("六九和弦 (1-3-5-6-9)", 4, (0, 4, 7, 9, 14)),
    ("属七降九 (1-3-5-b7-b9)", 4, (0, 4, 7, 10, 13)),
    ("属七升九 (1-3-5-b7-#9)", 4, (0, 4, 7, 10, 15)),
    ("属七升五 (1-3-#5-b7)", 3, (0, 4, 8, 10)),
    ("属七降五 (1-3-b5-b7)", 3, (0, 4, 6, 10)),
    # ---- 十一 / 十三 ----
    ("十一和弦 (1-3-5-b7-9-11)", 5, (0, 4, 7, 10, 14, 17)),
    ("十三和弦 (1-3-5-b7-9-13)", 5, (0, 4, 7, 10, 14, 21)),
    ("小十一和弦 (1-b3-5-b7-9-11)", 5, (0, 3, 7, 10, 14, 17)),
    # ---- 特殊 ----
    ("强力和弦 (1-5)", 1, (0, 7)),
    ("八度 (1-8)", 1, (0, 12)),
    ("三全音 (1-b5)", 1, (0, 6)),
)
CHORD_NONE = "无"
CUSTOM_CHORD_NAME = "自定义"
CUSTOM_CHORD_LO = 60                 # 自定义和弦窗的两个八度: C4..B5
CUSTOM_CHORD_HI = 83


def chord_intervals(intervals, inversion):
    """把和弦转位。

    间隔表按"从**低音**往上数"给出（预设就是这么写的，例如大三和弦 [0,4,7]
    表示低音之上的三个音）。

    转位 = 把最低的 n 个音各往上挪一个八度，其它音不动，再把整体平移回 0。
    **低音位置固定就是鼠标那个音**，转位只改变它上面的音程结构：

        大三 [0,4,7]
          原位    -> [0, 4, 7]   鼠标在 C -> C E G
          第一转位 -> [0, 5, 9]   鼠标在 C -> C F A
          第二转位 -> [0, 3, 8]   鼠标在 C -> C D# G#

    返回前去重：八度、强力和弦这种音数少的，转位后可能出现重复音。
    """
    try:
        iv = sorted(set(int(x) for x in intervals))
    except (TypeError, ValueError):
        return []           # 残缺的间隔表（手写配置等）直接当作关闭
    if not iv:
        return []
    inv = max(0, min(int(inversion), len(iv) - 1))
    for _ in range(inv):
        iv = sorted([iv[0] + 12] + iv[1:])
    base = iv[0]
    return sorted(set(x - base for x in iv))


def chord_note_set(bass_midi, intervals, inversion=0):
    """给定低音音高，算出实际要画的音（可能超过 MIDI_MAX，由调用方裁剪）。"""
    return [int(bass_midi) + d for d in chord_intervals(intervals, inversion)]


DB_FLOOR = -100.0
DISPLAY_GAMMA = 1.0

FFT_MIN = 1024
FFT_MAX = 262144
DEFAULT_TARGET_FPS = 150
FPS_CHOICES = (60, 100, 150, 200)

WINDOW_CHOICES = (
    "hann",
    "blackman-harris",
    "kaiser",
    "hamming",
    "flattop",
    "blackman-harris-7",
)
DEFAULT_WINDOW = "hann"

SPLAT_K = 3
SPLAT_SIGMA_LOW = 0.55
SPLAT_SIGMA_HIGH = 1.0

ENERGY_THRESHOLD_DB = -60.0

TIME_SPLAT_K = 2
TIME_SPLAT_SIGMA = 0.6
TIME_INV_SIGMA_SQ_HALF = -0.5 / (TIME_SPLAT_SIGMA * TIME_SPLAT_SIGMA)

GROUP_SEMITONES = 6
GROUP_OVERLAP = 4

F32 = np.float32
F64 = np.float64


# =========================================================================
# 颜色映射
# =========================================================================
MAGMA_STOPS = [
    (0, 0, 4),
    (28, 16, 68),
    (79, 18, 123),
    (129, 37, 129),
    (181, 54, 122),
    (229, 80, 100),
    (251, 135, 97),
    (254, 194, 135),
    (252, 253, 191),
]
INFERNO_STOPS = [
    (0, 0, 4),
    (22, 11, 57),
    (66, 10, 104),
    (106, 23, 110),
    (147, 38, 103),
    (188, 55, 84),
    (221, 81, 58),
    (243, 120, 25),
    (252, 165, 10),
    (246, 215, 70),
    (252, 255, 164),
]
VIRIDIS_STOPS = [
    (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (109, 205, 89),
    (180, 222, 44),
    (253, 231, 37),
]
ICE_STOPS = [
    (2, 4, 10),
    (8, 22, 50),
    (10, 52, 96),
    (8, 92, 140),
    (20, 140, 175),
    (70, 185, 205),
    (140, 220, 230),
    (210, 245, 250),
    (255, 255, 255),
]


def build_lut(stops, n=256):
    stops = np.asarray(stops, dtype=np.float64) / 255.0
    x = np.linspace(0.0, 1.0, len(stops))
    xi = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(xi, x, stops[:, c]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lut[0] = (0, 0, 0)
    return lut


LUTS = {
    "magma": build_lut(MAGMA_STOPS),
    "inferno": build_lut(INFERNO_STOPS),
    "viridis": build_lut(VIRIDIS_STOPS),
    "ice": build_lut(ICE_STOPS),
}
LUTS_RGB = {name: (lut[:, 0].copy(), lut[:, 1].copy(), lut[:, 2].copy()) for name, lut in LUTS.items()}


# =========================================================================
# MIDI / 频率 / 坐标
# =========================================================================
def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def midi_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def midi_to_y(m, height, midi_min=MIDI_MIN, midi_max=MIDI_MAX):
    span = midi_max - midi_min + 1
    return (midi_max + 0.5 - m) / span * height


def y_to_midi(y, height, midi_min=MIDI_MIN, midi_max=MIDI_MAX):
    span = midi_max - midi_min + 1
    return (midi_max + 0.5) - (y / height) * span


# =========================================================================
# 音频加载 / 编码
# =========================================================================
def _to_float(data):
    data = np.asarray(data)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    return data


def to_mono(samples):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        return np.ascontiguousarray(x)
    if x.ndim == 2:
        if x.shape[1] == 1:
            return np.ascontiguousarray(x[:, 0])
        return np.ascontiguousarray(x.mean(axis=1))
    raise ValueError(f"不支持的采样维度: {x.ndim}")


def load_audio(path):
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            return np.ascontiguousarray(data[:, 0]), sr
        return np.ascontiguousarray(data), sr
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from scipy.io import wavfile
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sr, data = wavfile.read(path)
        data = _to_float(data)
        if data.ndim == 1:
            return np.ascontiguousarray(data), sr
        if data.shape[1] == 1:
            return np.ascontiguousarray(data[:, 0]), sr
        return np.ascontiguousarray(data), sr
    except ImportError:
        pass
    except Exception:
        pass

    try:
        import wave as _wave

        with _wave.open(path, "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif sw == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"不支持的位深: {sw*8} bit")
        if nch > 1:
            arr = arr.reshape(-1, nch)
        return np.ascontiguousarray(arr), sr
    except Exception as e:
        raise RuntimeError("无法解码该音频文件。建议 pip install soundfile\n" f"原始错误: {e}")


def samples_to_wav_bytes(samples, sr):
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    elif x.ndim == 2:
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        elif x.shape[1] > 2:
            x = x[:, :2]
    else:
        raise ValueError(f"不支持的采样维度: {x.ndim}")

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, -1.0, 1.0, out=x)
    pcm = np.round(x * 32767.0).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# =========================================================================
# 窗函数
# =========================================================================
_WINDOW_CACHE = {}


def make_window(name, N):
    """生成窗函数（结果带缓存）。

    统一使用**周期型（DFT-even）**形式，即余弦分母取 N。
    这样窗在 STFT 里首尾相接、不需要额外补一个样本；混用周期型和对称型
    （分母 N-1）会让不同窗的对比结果带上非预期差异。

    N=1 时周期型 Hann 等窗会退化成全零（cos(0) 项互相抵消），
    因此 N<2 一律返回全 1 的矩形窗。
    """
    key = (name, int(N))
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached

    if N < 2:
        w = np.ones(max(0, int(N)), dtype=np.float32)
        w = np.ascontiguousarray(w)
        _WINDOW_CACHE[key] = w
        return w

    n = np.arange(N, dtype=np.float64)
    if name == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / N)
    elif name == "blackman-harris":
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        w = (a0
             - a1 * np.cos(2.0 * np.pi * n / N)
             + a2 * np.cos(4.0 * np.pi * n / N)
             - a3 * np.cos(6.0 * np.pi * n / N))
    elif name == "blackman-harris-7":
        a = (0.27105140069342, 0.43329793923448, 0.21812299954311,
             0.06592544638803, 0.01081174209837, 0.00077658482522,
             0.00001388721735)
        w = np.full(N, a[0], dtype=np.float64)
        for i in range(1, 7):
            sign = -1.0 if (i % 2 == 1) else 1.0
            w += sign * a[i] * np.cos(2.0 * np.pi * i * n / N)
    elif name == "kaiser":
        beta = 14.0
        x = (2.0 * n / N) - 1.0
        w = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - x * x))) / np.i0(beta)
    elif name == "flattop":
        a = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
        w = (a[0]
             - a[1] * np.cos(2.0 * np.pi * n / N)
             + a[2] * np.cos(4.0 * np.pi * n / N)
             - a[3] * np.cos(6.0 * np.pi * n / N)
             + a[4] * np.cos(8.0 * np.pi * n / N))
    else:
        # hann 及未知名一律走周期型 Hann
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / N)

    w = np.ascontiguousarray(w.astype(np.float32))
    _WINDOW_CACHE[key] = w
    return w


# =========================================================================
# FFT 分组规划
# =========================================================================
def _next_pow2(n):
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _prev_pow2(n):
    n = max(1, int(n))
    return 1 << (n.bit_length() - 1)


def _plan_groups(sr, midi_min, midi_max, min_fft=FFT_MIN, max_fft=FFT_MAX, group_semitones=GROUP_SEMITONES, overlap=GROUP_OVERLAP, n_samples=None):
    """按半音分组，为每组选 FFT 尺寸 N。

    两条约束：

    1. N 不能超过信号长度 n_samples。
       窗比信号还长时，窗里只有一小截真实数据、其余是补出来的样本，
       相位差法测出的瞬时频率随之失效，整个低频段会糊掉。把 N 限制在
       信号长度以内，可以保证窗内始终是真实数据。

    2. 同一八度内的相邻两组（A-D 与 D#-G#）必须用同一个 N。
       每组的分辨率需求由它**最低音**的半音间距决定，而 D# 比 A 高一个
       三全音，于是 D#-G# 组的 bin 宽恰好是配对 A-D 组的 2 倍 —— 纵向分辨率
       粗一倍，能量在更多行之间摊开，画出来就是 D#-G# 区域明显更稀疏。
       统一 N 之后这个不对称消失。
    """
    ratio = 2.0 ** (1.0 / 12.0) - 1.0
    if n_samples is not None:
        # 至少要能放下 min_fft，否则这组没法分析；
        # 同时保持 2 的幂（用 _prev_pow2 向下取），否则会出现非 2 幂的
        # FFT 长度，且和相邻组的分辨率对不齐。
        cap = _prev_pow2(max(1, int(n_samples)))
        max_fft = int(max(min_fft, min(max_fft, cap)))
    groups = []
    m = midi_min
    while m <= midi_max:
        m_end = min(m + group_semitones, midi_max + 1)
        delta_f = midi_to_freq(m) * ratio
        N = _next_pow2(6.0 * sr / delta_f)
        N = int(min(max(N, min_fft), max_fft))
        hop_nat = max(1, _prev_pow2(max(1, N // overlap)))
        # hop 超过窗长会漏掉样本，必须夹住
        hop_nat = int(min(hop_nat, max(1, N)))
        groups.append([m, m_end, N, hop_nat])
        m = m_end

    # 同一八度对内的两组统一分辨率：取两者中更细的 N
    i = 0
    while i + 1 < len(groups):
        a, b = groups[i], groups[i + 1]
        if b[0] - a[0] == group_semitones:
            N = max(a[2], b[2])
            for g in (a, b):
                g[2] = N
                # 窗变长后 hop 也要按同样的重叠比重新夹一次
                h = max(1, _prev_pow2(max(1, N // overlap)))
                g[3] = int(min(max(h, min(g[3], h)), N))
            i += 2
        else:
            i += 1
    return [tuple(g) for g in groups]


def _midpoint_freq(f_a, f_b):
    """两个频率的几何中点（音乐上是它们之间的半音中点）。"""
    return math.sqrt(float(f_a) * float(f_b))


def _plan_chunks(sr, n_samples, groups):
    """为每组规划 bin 范围与分块。

    相邻组的频率区间必须**精确分割**，切分点取两组交界处的几何中点。
    不能各自用 searchsorted 去够自己的标称边界：bin 分辨率有限（最低几组
    一个 bin 才 0.17 Hz 量级），那样会让相邻两组同时包含边界处的那个 bin，
    于是正好落在分组边界上的音（例如 A4，它是 A4-D5 组的首音）被两组各算
    一遍、能量翻倍，画出来是每 6 个半音出现一次的亮度条纹。

    每个元组末尾额外带出 (f_lo, f_hi)，供 _reassign_chunk 判断重分配后的
    频率是否仍属于本组 —— 输入 bin 与输出频带用同一套边界，两者才一致。
    """
    n_groups = len(groups)
    # 每组的下边界：与本组前一组的几何中点；最低一组从 0 开始
    band_lo = []
    for i, (m_start, m_end, N, hop_g) in enumerate(groups):
        if i == 0:
            band_lo.append(0.0)
        else:
            band_lo.append(_midpoint_freq(midi_to_freq(groups[i - 1][1] - 1), midi_to_freq(m_start)))
    # 每组的上边界：与下一组的几何中点；最高一组到 Nyquist
    band_hi = []
    for i, (m_start, m_end, N, hop_g) in enumerate(groups):
        if i == n_groups - 1:
            band_hi.append(float(sr) * 0.5)
        else:
            band_hi.append(_midpoint_freq(midi_to_freq(m_end - 1), midi_to_freq(groups[i + 1][0])))

    plans = []
    for i, (m_start, m_end, N, hop_g) in enumerate(groups):
        n_frames_g = n_samples // hop_g + 1
        n_bins = N // 2 + 1
        f_k_np = np.arange(n_bins, dtype=np.float64) * (float(sr) / float(N))

        f_lo = band_lo[i]
        f_hi = band_hi[i]
        k_lo = int(np.searchsorted(f_k_np, f_lo, side="left"))
        k_hi = int(np.searchsorted(f_k_np, f_hi, side="left"))
        k_lo = max(0, min(k_lo, n_bins - 1))
        k_hi = max(k_lo + 1, min(k_hi, n_bins))
        # 上面的钳制可能让 k_lo 落回本组频带之外（最低几组 bin 很粗），
        # 那会把属于上一组的 bin 又拉进来，重新造成重叠。此时直接判定为空。
        if f_k_np[k_lo] >= f_hi:
            k_lo = k_hi = 0

        target_mem = 1.5e8 if HAS_GPU else 4.0e8
        chunk = max(8, int(target_mem / (n_bins * 16 * 3)))
        starts = []
        i0 = 0
        while i0 < n_frames_g - 2:
            i1 = min(i0 + chunk, n_frames_g)
            if i1 - i0 < 3:
                break
            starts.append((i0, i1))
            if i1 >= n_frames_g:
                break
            i0 = i1 - 2

        plans.append((m_start, m_end, N, hop_g, k_lo, k_hi, starts, f_lo, f_hi))
    return plans


# =========================================================================
# 取消异常
# =========================================================================
class AnalysisCancelled(Exception):
    pass


# =========================================================================
# STFT（单路：只做 X = FFT(x·w)）
# =========================================================================
def _stft_batch(samples_xp, N, hop, i0, i1, window_xp):
    n = int(samples_xp.shape[0])
    half = N // 2

    starts = xp.arange(i0, i1, dtype=xp.int64) * hop - half
    base = xp.arange(N, dtype=xp.int64)
    idx = starts[:, None] + base[None, :]

    valid = (idx >= 0) & (idx < n)
    xp.clip(idx, 0, n - 1, out=idx)

    frames = samples_xp[idx]
    frames *= valid
    frames *= window_xp[None, :]

    X = xp.fft.rfft(frames, axis=1)
    X *= F32(1.0 / float(N))
    return X


# =========================================================================
# 谱重分配：帧间相位差求瞬时频率，再按频率把能量搬到对应半音行
# =========================================================================
def _reassign_chunk(
    X,
    N,
    hop,
    sr,
    f_k,
    n_rows,
    midi_min,
    midi_max,
    rows_per_semitone,
    group_grid,
    i0_group,
    k_lo=0,
    k_hi=None,
    band_lo=None,
    band_hi=None,
):
    m_full, n_bins_full = X.shape
    if m_full < 3:
        return
    if k_hi is None:
        k_hi = n_bins_full

    X_sub = X[:, k_lo:k_hi]
    m, nb = X_sub.shape
    if m < 3 or nb == 0:
        return

    f_k_sub = f_k[k_lo:k_hi]

    mag = xp.abs(X_sub)
    phase = xp.angle(X_sub)

    # 三帧相位差
    dphase = (phase[2:] - phase[:-2]) * 0.5
    del phase

    mag_mid = mag[1:-1]
    del mag

    k_idx = xp.arange(k_lo, k_hi, dtype=xp.float64)
    # 减掉每个 bin 在 hop 之间的固有相位推进，再折算成频率偏差。
    # 折到 (-pi, pi] 的写法：((x + pi) mod 2pi) - pi。
    dphase -= (2.0 * xp.pi * float(hop) / float(N)) * k_idx[None, :]
    dphase = xp.mod(dphase + xp.pi, 2.0 * xp.pi) - xp.pi
    dphase *= float(sr) / (2.0 * xp.pi * float(hop))

    f_inst = f_k_sub[None, :] + dphase
    del dphase
    f_inst = xp.maximum(f_inst, F64(1e-9))

    m_midi = 69.0 + 12.0 * xp.log2(f_inst * (1.0 / 440.0))
    r_float = (float(midi_max) + 0.5 - m_midi) * float(rows_per_semitone) - 0.5
    del m_midi

    r_lo = -float(SPLAT_K) - 1.0
    r_hi = float(n_rows) + float(SPLAT_K)
    valid = (f_inst > 20.0) & (f_inst < float(sr) * 0.5) & (r_float > r_lo) & (r_float < r_hi)

    # 频带归属：只累加重分配后仍落在本组频带内的能量。
    # 输入 bin 范围与这里的频带用同一套边界（见 _plan_chunks），
    # 半开区间 [band_lo, band_hi) 保证相邻组不重不漏。
    if band_lo is not None and band_hi is not None:
        valid &= (f_inst >= band_lo) & (f_inst < band_hi)
    del f_inst

    if ENERGY_THRESHOLD_DB < 0.0:
        frame_max = mag_mid.max(axis=1, keepdims=True)
        frame_max = xp.maximum(frame_max, F32(1e-12))
        threshold = frame_max * F32(10.0 ** (ENERGY_THRESHOLD_DB / 20.0))
        valid = valid & (mag_mid > threshold)
        del frame_max, threshold

    tv, kv = xp.where(valid)
    del valid
    if tv.size == 0:
        return

    r_v = r_float[tv, kv]
    mag_v = mag_mid[tv, kv]
    del r_float, mag_mid

    # σ 随频率变化：低频窄（SPLAT_SIGMA_LOW）→ 高频宽（SPLAT_SIGMA_HIGH）
    m_midi_v = (float(midi_max) + 0.5) - (r_v + 0.5) / float(rows_per_semitone)
    span_midi = float(midi_max - midi_min) if midi_max > midi_min else 1.0
    t_norm = (m_midi_v - float(midi_min)) / span_midi
    t_norm = xp.clip(t_norm, 0.0, 1.0).astype(F32)
    sigma_v = SPLAT_SIGMA_LOW + (SPLAT_SIGMA_HIGH - SPLAT_SIGMA_LOW) * t_norm
    inv_sigma_sq_half_v = (-0.5 / (sigma_v * sigma_v)).astype(F32)
    del m_midi_v, t_norm, sigma_v

    r_base = xp.floor(r_v).astype(xp.int64)
    frac = (r_v - r_base).astype(F32)
    del r_v

    global_t = tv + 1
    t_lo = int(global_t.min())
    t_hi = int(global_t.max()) + 1
    span = t_hi - t_lo
    if span <= 0:
        return
    t_shift = global_t - t_lo
    size = span * n_rows

    for off in range(-SPLAT_K, SPLAT_K + 1):
        r_target = r_base + off
        d = frac - off
        w = xp.exp(inv_sigma_sq_half_v * d * d)
        del d

        ok = (r_target >= 0) & (r_target < n_rows)
        if not bool(ok.any()):
            continue

        t_ok = t_shift[ok]
        r_ok = r_target[ok]
        v_ok = mag_v[ok] * w[ok]
        flat = t_ok * n_rows + r_ok

        contrib = xp.bincount(flat, weights=v_ok, minlength=size).astype(F32)
        group_grid[i0_group + t_lo : i0_group + t_hi] += contrib.reshape(span, n_rows)


# =========================================================================
# 群组网格 → 公共网格：时间方向的高斯重采样
# =========================================================================
def _splat_group_to_common(group_grid, ratio, out, n_common):
    n_g, n_rows = group_grid.shape
    if n_g <= 0:
        return

    if ratio == 1:
        m = min(n_g, n_common)
        if m > 0:
            out[:m] += group_grid[:m]
        return

    t_arr = xp.arange(n_common, dtype=xp.float64) / float(ratio)
    i0 = xp.floor(t_arr).astype(xp.int64)
    frac = (t_arr - i0).astype(F32)

    for off in range(-TIME_SPLAT_K, TIME_SPLAT_K + 1):
        idx = i0 + off
        valid = (idx >= 0) & (idx < n_g)
        if not bool(valid.any()):
            continue

        d = frac - off
        w = xp.exp(TIME_INV_SIGMA_SQ_HALF * d * d).astype(F32)
        w = w * valid
        if not bool((w > 1e-6).any()):
            continue

        idx_clipped = xp.clip(idx, 0, n_g - 1)
        out += group_grid[idx_clipped] * w[:, None]


# =========================================================================
# 主流程
# =========================================================================
def compute_reassigned_spectrogram(
    samples,
    sr,
    midi_min=MIDI_MIN,
    midi_max=MIDI_MAX,
    rows_per_semitone=DEFAULT_ROWS_PER_SEMITONE,
    target_fps=DEFAULT_TARGET_FPS,
    window_name=DEFAULT_WINDOW,
    max_frames=65536,
    progress_cb=None,
    cancel_cb=None,
):
    """公开入口。持有 XP_LOCK，保证后台 GPU 探测不会在分析中途切换后端。"""
    # 等后端确定下来，但最多等 _GPU_PROBE_WAIT 秒：探测卡住也照样能用 CPU 出图。
    _ensure_backend_ready()
    with XP_LOCK:
        return _compute_reassigned_locked(
            samples,
            sr,
            midi_min,
            midi_max,
            rows_per_semitone,
            target_fps,
            window_name,
            max_frames,
            progress_cb,
            cancel_cb,
        )


def _compute_reassigned_locked(
    samples,
    sr,
    midi_min,
    midi_max,
    rows_per_semitone,
    target_fps,
    window_name,
    max_frames,
    progress_cb,
    cancel_cb,
):
    n_rows = (midi_max - midi_min + 1) * rows_per_semitone
    samples_np = np.ascontiguousarray(samples, dtype=F32)
    n = len(samples_np)

    if n == 0:
        return np.zeros((1, n_rows), dtype=F32), max(1, sr // target_fps)

    # 窗长不能超过信号本身，否则窗只是一小截真实数据（见 _plan_groups 注释）
    groups_raw = _plan_groups(sr, midi_min, midi_max, n_samples=n)

    target_hop = max(1, int(round(sr / float(target_fps))))
    base_hop = max(1, _prev_pow2(target_hop))
    # max_frames 至少为 2，否则下面的 (max_frames - 1) 会除零
    max_frames = max(2, _int_opt(max_frames, 65536))
    if n // base_hop + 1 > max_frames:
        needed = int(math.ceil(n / float(max_frames - 1)))
        base_hop = max(base_hop, _next_pow2(needed))
    n_common = max(2, n // base_hop + 1)

    groups = [(ms, me, N, max(base_hop, hop_nat)) for (ms, me, N, hop_nat) in groups_raw]

    samples_xp = xp.asarray(samples_np)
    out_xp = xp.zeros((n_common, n_rows), dtype=F32)

    window_cache = {}

    def get_window(N):
        w = window_cache.get(N)
        if w is None:
            w = xp.asarray(make_window(window_name, N))
            window_cache[N] = w
        return w

    plans = _plan_chunks(sr, n, groups)
    total_chunks = sum(len(p[6]) for p in plans) or 1
    done_chunks = 0

    # 相位差至少要三帧才能算，帧数不足的组会被直接跳过。
    # 另外，短信号会把最长 FFT 压小，某些组的频率分辨率可能已经差到
    # 分不开自己负责的半音 —— 这时低音区必然糊。两种情况都给一次明确告警，
    # 避免"静默出全黑图/低频发虚"这种难查的现象。
    if plans:
        min_frames = min(n // hop + 1 for (_, _, _, hop, _, _, _, _, _) in plans)
        if min_frames < 3:
            print(f"[analyze] 信号过短（{n} 个采样 ≈ {n / float(sr) * 1000:.1f} ms），" f"最少只有 {min_frames} 帧，不足 3 帧，结果将为空。")
        else:
            worst = None
            for ms, me, N, hop_g, _kl, _kh, _st, _fl, _fh in plans:
                df = float(sr) / float(N)
                semitone = midi_to_freq(ms + 1) - midi_to_freq(ms)
                if df > 2.0 * semitone:
                    if worst is None or df / semitone > worst[2]:
                        worst = (ms, df, df / semitone)
            if worst is not None:
                ms, df, ratio = worst
                print(f"[analyze] 信号偏短（{n / float(sr):.2f} s）：最长 FFT 受信号长度" f"限制，{midi_name(ms)} 附近分辨率 " f"{df:.1f} Hz 已达半音间距的 {ratio:.1f} 倍，低音区会明显发糊。")

    if progress_cb is not None:
        try:
            progress_cb(0.0)
        except Exception:
            pass

    cancelled = False
    for m_start, m_end, N, hop_g, k_lo, k_hi, starts, band_lo, band_hi in plans:
        if not starts:
            continue

        ratio = max(1, int(hop_g // base_hop))
        n_g_frames = n // hop_g + 1

        group_grid = xp.zeros((n_g_frames, n_rows), dtype=F32)
        window_xp = get_window(N)
        n_bins = N // 2 + 1
        f_k_xp = xp.arange(n_bins, dtype=xp.float64) * (float(sr) / float(N))

        for i0, i1 in starts:
            if cancel_cb is not None:
                try:
                    if cancel_cb():
                        cancelled = True
                        break
                except Exception:
                    pass

            X = _stft_batch(samples_xp, N, hop_g, i0, i1, window_xp)
            _reassign_chunk(
                X,
                N,
                hop_g,
                sr,
                f_k_xp,
                n_rows,
                midi_min,
                midi_max,
                rows_per_semitone,
                group_grid,
                i0,
                k_lo=k_lo,
                k_hi=k_hi,
                band_lo=band_lo,
                band_hi=band_hi,
            )
            del X

            done_chunks += 1
            if progress_cb is not None:
                try:
                    progress_cb(done_chunks / total_chunks)
                except Exception:
                    pass

        if cancelled:
            break

        _splat_group_to_common(group_grid, ratio, out_xp, n_common)
        del group_grid

    if cancelled:
        raise AnalysisCancelled()

    if HAS_GPU:
        out_np = xp.asnumpy(out_xp).astype(F32, copy=False)
    else:
        out_np = np.asarray(out_xp, dtype=F32)

    return out_np, base_hop


# =========================================================================
# dB / 显示
# =========================================================================
def compute_db(mag, floor_db=DB_FLOOR):
    if mag.size == 0:
        return mag
    mx = float(mag.max())
    if not np.isfinite(mx) or mx <= 1e-20:
        return np.full(mag.shape, floor_db, dtype=F32)

    db = np.empty_like(mag, dtype=F32)
    np.maximum(mag, F32(1e-12), out=db)
    np.log10(db, out=db)
    db *= F32(20.0)
    db -= F32(float(db.max()))
    np.clip(db, floor_db, 0.0, out=db)
    return db


def db_to_u8(db, floor_db=DB_FLOOR, hide_low=0.0, gamma=DISPLAY_GAMMA):
    span = -float(floor_db)
    norm = np.empty_like(db, dtype=F32)
    np.subtract(db, F32(floor_db), out=norm)
    norm *= F32(1.0 / span)
    if hide_low > 0.0:
        inv = 1.0 / max(1.0 - hide_low, 1e-9)
        norm -= F32(hide_low)
        np.maximum(norm, F32(0.0), out=norm)
        norm *= F32(inv)
    if gamma != 1.0:
        np.power(norm, F32(gamma), out=norm)
    np.clip(norm, F32(0.0), F32(1.0), out=norm)
    norm *= F32(255.0)
    norm += F32(0.5)
    return norm.astype(np.uint8)


# =========================================================================
# 阶梯滤镜：把每个半音压成一条平带，制造半音之间的硬台阶
# -------------------------------------------------------------------------
# 分两步：
#   1. 归约 —— 一个半音占 rows_per_semitone 行，先压成一个特征值。
#      mean   : 平均。会让半音内变平，但中心峰值被摊薄。
#      midmax : 去掉最上/最下各 N 行后取最大。峰值通常在半音中间、
#               而扩散到边界的能量堆在两头，所以这样既保住峰值又丢掉泄漏。
#   2. 曲线 —— 可选。把弱半音继续压低、强半音基本不动，让台阶更陡。
#               四种曲线都只压不强推，不做任何提亮。
# 只作用于显示，不改动分析结果和导出的数据。
# =========================================================================
CURVE_MODES = (
    "none",      # 不压
    "knee",      # 低于阈值的部分乘以增益
    "power",     # 以 pivot 为支点的幂曲线
    "sigmoid",   # 以 center 为中点的 S 曲线
)

DEFAULT_FILTER_CONFIG = {
    "reduce": "midmax",   # mean | midmax
    "trim": 3,            # midmax 去掉的上下行数；0 等同于整段取最大
    "curve": "none",      # 见 CURVE_MODES
    "knee_th": 0.15,      # knee 阈值（线性幅度）
    "knee_gain": 0.5,     # knee 以下乘这个增益
    "pow_gamma": 2.0,     # power 的 gamma（>1 压暗）
    "pow_pivot": 0.30,    # power 的支点
    "sig_center": 0.30,   # sigmoid 拐点，占显示范围的比例（1.0 = DB_FLOOR）
    "sig_k": 0.50,        # sigmoid 过渡宽度，占拐点深度的比例
}


def _num_opt(value, default):
    """把配置里的数值转成 float；None/垃圾值一律退回默认值。"""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_opt(value, default):
    """把配置里的数值转成 int；None/垃圾值一律退回默认值。"""
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _filter_config(cfg=None):
    """合并用户配置与默认值。

    未在默认配置里的键直接忽略；值为 None 或无法转成数值的也退回默认，
    这样残缺的配置（手写文件、旧版本留下的键）不会让滤镜抛异常。
    """
    out = dict(DEFAULT_FILTER_CONFIG)
    if cfg:
        out.update({k: v for k, v in cfg.items()
                    if k in out and v is not None})
    for key in ("knee_th", "knee_gain", "pow_gamma", "pow_pivot",
                "sig_center", "sig_k"):
        out[key] = _num_opt(out.get(key), DEFAULT_FILTER_CONFIG[key])
    out["trim"] = _int_opt(out.get("trim"), DEFAULT_FILTER_CONFIG["trim"])
    if out.get("reduce") not in ("mean", "midmax"):
        out["reduce"] = DEFAULT_FILTER_CONFIG["reduce"]
    if out.get("curve") not in CURVE_MODES:
        out["curve"] = DEFAULT_FILTER_CONFIG["curve"]
    return out


def _apply_curve(mag_db, c):
    """对归约后的 dB 施加所选对比曲线，返回同形状的 dB。

    统一约束：曲线**只压不强推**，且必须**单调不减** —— 输入更亮的结果不能
    更暗，输入更暗的结果不能更亮。否则画面上会出现反直觉的明暗倒挂。
    三条曲线都满足 out <= in 且 out(0 dB) == 0 dB（峰值不受影响）。

    knee   : 线性幅度低于阈值的部分乘以增益 g。
    power  : 以 pivot 为支点，pivot 以下按 gamma 压缩，以上保持原值。
    sigmoid: dB 域 S 形软限幅。以 center 为拐点，比它暗的部分被压向显示下限、
             比它亮的部分保持原值，中间是平滑过渡（不会像 knee 那样在阈值处
             留一条硬折线）。
    """
    mode = c["curve"]
    if mode == "knee":
        th = max(1e-9, float(c["knee_th"]))
        g = min(1.0, max(0.0, float(c["knee_gain"])))
        db_cut = 20.0 * np.log10(th)
        # 直接缩放到 g 倍：对 dB 而言就是整体平移 20lg(g)，天然单调
        return np.where(mag_db < db_cut, mag_db + 20.0 * np.log10(max(g, 1e-9)),
                        mag_db)
    if mode == "power":
        # gamma<=1 会把支点以下的值抬起来（提亮），与"只压不强推"矛盾，
        # 所以下限钳到 1：1 表示不压。
        gamma = max(1.0, float(c["pow_gamma"]))
        pivot = max(1e-9, float(c["pow_pivot"]))
        # 支点最高只能到 0 dB（峰值）；pivot=1.0 时浮点误差会算出 +1e-15，
        # 那会让"支点以上保持原值"的分支失效，所以这里显式夹住。
        db_pivot = min(0.0, 20.0 * np.log10(pivot))
        # 支点以下按 gamma 压缩；支点以上不动（否则会被放大而过曝）
        return np.where(mag_db < db_pivot,
                        db_pivot + (mag_db - db_pivot) * gamma, mag_db)
    if mode == "sigmoid":
        # dB 域平滑软限幅。以拐点 center 为界：
        #   d >= center+half : 原样保留（亮部不动）
        #   d ~= center      : 压掉 50%
        #   d <= center-half : 压掉 100%（= 直接归到显示下限）
        # 压掉的量以 dB 计，与输入电平无关，所以"压一半"就是字面意义的
        # 再低 20·lg(0.5) dB，不会出现越暗被压得越狠的失控。
        # g 单调递增 => 输出单调不减；loss <= -d => 输出不低于显示下限。
        frac = min(1.0, max(0.0, float(c["sig_center"])))
        if frac <= 0.0:
            return mag_db
        floor_db = abs(float(DB_FLOOR))
        center = -floor_db * frac                   # 拐点（dB，负数）
        # sig_k 是过渡半宽，按拐点深度取比例；太窄会退化成硬阈值，钳到 >= 1 dB
        k = min(1.0, max(0.0, float(c["sig_k"])))
        half = max(1.0, k * frac * floor_db * 0.5)
        g = 0.5 * (1.0 + np.tanh((mag_db - center) / half))     # 0..1，单调
        loss = -mag_db * (1.0 - g)                  # 要压掉的 dB 数（>= 0）
        return np.maximum(mag_db - np.minimum(loss, -mag_db), float(DB_FLOOR))
    return mag_db


# midmax 的 trim 预设：以"12 行去掉上下各 3 行"为基准的比例表。
# 纯比例在 6 行时会算出 1.5（四舍五入成 2），实测 1 更合适，所以直接列出来。
TRIM_PRESETS = {1: 0, 2: 1, 3: 1, 4: 1, 6: 1, 8: 2, 12: 3, 16: 4, 24: 6, 32: 8, 48: 12}


def scaled_trim(trim, rows_per_semitone):
    """把按 12 行调好的 trim 折算到当前每半音行数。

    基准：12 行去掉上下各 3 行（保留中间 6 行）。行数变了要按比例缩放，
    否则视觉权重会变。常见行数直接用预设表，其它行数按比例四舍五入。

    rows_per_semitone <= 2 时 midmax 没有"中间行"可留（去任何一行都会空），
    统一返回 0，也就是退化成整段取最大。

    注意：只要 trim > 0 且行数够，结果至少是 **1** —— 折算成 0 等于把
    "去掉边界"悄悄关掉，用户会以为滤镜坏了。
    """
    rps = max(1, int(rows_per_semitone))
    if rps <= 2:
        return 0
    base = int(TRIM_PRESETS.get(12, 3))
    if int(trim) == base and rps in TRIM_PRESETS:
        return max(0, min(TRIM_PRESETS[rps], (rps - 1) // 2))
    got = int(round(float(trim) * rps / 12.0))
    if int(trim) > 0:
        got = max(1, got)
    return max(0, min(got, (rps - 1) // 2))


def _reduce_blocks(blocks, c):
    """把一个半音内的多行压成一个特征值。

    blocks: (frames, n_semi, rps) 的 dB → (frames, n_semi) 的 dB
    """
    rps = blocks.shape[2]
    if c["reduce"] == "mean":
        # 在 dB 域取平均，更接近视觉上的等量
        return blocks.mean(axis=2, dtype=F64)
    # midmax：按当前行数折算 trim，去掉上下各 N 行后取最大
    trim = scaled_trim(c["trim"], rps)
    if trim <= 0:
        return blocks.max(axis=2)
    return blocks[:, :, trim:rps - trim].max(axis=2)


def semitone_bands(db, rows_per_semitone=DEFAULT_ROWS_PER_SEMITONE,
                   midi_max=MIDI_MAX, config=None):
    """把每个半音压成一个特征值，返回 (n_frames, n_semi) 的 dB，**不做铺开**。

    这是阶梯滤镜的核心：铺开成每个半音 rps 行的那一步（np.repeat）纯属浪费 ——
    铺开以后同一半音内的 rps 行完全一样，上色和绘制时会被重复处理 rps 遍。
    显示端直接按"一个半音一行"去上色，再让 QPainter 纵向拉伸补满，
    因为每个半音占的行数相同，拉伸后的位置与逐行铺开完全一致。
    """
    c = _filter_config(config)
    rps = max(1, int(rows_per_semitone))
    n_frames, n_rows = db.shape
    n_semi = n_rows // rps
    if n_semi <= 0:
        return db.reshape(n_frames, 0) if n_rows == 0 else db
    blocks = db[:, :n_semi * rps].reshape(n_frames, n_semi, rps)
    red = _reduce_blocks(blocks, c)                     # (frames, n_semi) dB
    return _apply_curve(red, c)                         # dB 域过对比曲线


def semitone_step_filter(db, rows_per_semitone=DEFAULT_ROWS_PER_SEMITONE,
                         midi_max=MIDI_MAX, config=None):
    """阶梯滤镜：把每个半音压成一条平带，制造半音之间的硬台阶。

    先按 semitone_bands 归约 + 过曲线，再把结果铺回该半音的每一行 ——
    于是半音内部完全均匀、半音边界出现跳变。

    输入/输出都是 (n_frames, n_rows) 的 dB，行 0 对应 MIDI_MAX。
    只影响显示，分析结果与导出数据都不变。
    """
    rps = max(1, int(rows_per_semitone))
    n_frames, n_rows = db.shape
    if n_rows <= rps:
        return db
    flat = semitone_bands(db, rps, midi_max, config)
    n_semi = flat.shape[1]
    used = n_semi * rps
    out = np.empty_like(db)
    out[:, :used] = np.repeat(flat[:, :, None], rps, axis=2).reshape(n_frames, used)
    # 末尾不足一个半音的行按最后一个完整半音处理
    if used < n_rows:
        out[:, used:] = flat[:, -1:]
    return out


def u8_to_qimage_fast(u8, rgb_lut):
    """u8: (n_frames, n_rows) → QImage: width=n_frames, height=n_rows

    行 0 对应 MIDI_MAX（最高音），所以输出图的行序与数组行序一致。
    末尾 .copy() 是必须的：QImage 只引用我们传进去的缓冲区，不接管所有权。
    """
    r_lut, g_lut, b_lut = rgb_lut
    n_frames, n_rows = u8.shape
    rgb = np.empty((n_rows, n_frames, 3), dtype=np.uint8)
    rgb[:, :, 0] = r_lut[u8].T
    rgb[:, :, 1] = g_lut[u8].T
    rgb[:, :, 2] = b_lut[u8].T
    h, w = n_rows, n_frames
    img = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return img.copy()


# =========================================================================
# 分析结果导出（数组，而不是图片）
# =========================================================================
# 导出的是**原始线性幅度**，不做任何归一化/量化/裁剪 —— 调试和二次处理
# 需要的就是未经显示管线污染的数据。
EXPORT_FORMATS = ("npz", "npy", "csv", "raw(f32)")


def export_payload(mag, db, hop, sr, params=None, source_path=None,
                   hide_low=None, colormap=None, u8=None):
    """组装一份自描述的导出数据。

    mag : (n_frames, n_rows) float32，线性幅度，行 0 = MIDI_MAX
    db  : 同一形状的 dB（相对峰值，上限 0），便于直接画图
    """
    p = params or {}
    n_frames, n_rows = mag.shape
    midi_min = MIDI_MIN
    midi_max = MIDI_MAX
    rps = int(p.get("rows_per_semitone", DEFAULT_ROWS_PER_SEMITONE))
    frames = np.arange(n_frames, dtype=np.float64)
    times = frames * float(hop) / float(sr)
    rows = np.arange(n_rows)
    midis = (midi_max + 0.5) - (rows + 0.5) / float(rps)
    freqs = 440.0 * np.power(2.0, (midis - 69.0) / 12.0)
    return {
        "mag": np.asarray(mag, dtype=np.float32),
        "db": np.asarray(db, dtype=np.float32),
        "u8": (None if u8 is None else np.asarray(u8, dtype=np.uint8)),
        "time_s": times,
        "midi": midis,
        "freq_hz": freqs,
        # ---- 元数据（npz 里存成 0 维数组，csv 里写注释头）----
        "sr": int(sr),
        "hop": int(hop),
        "fps": float(sr) / float(hop),
        "midi_min": int(midi_min),
        "midi_max": int(midi_max),
        "rows_per_semitone": int(rps),
        "window_name": str(p.get("window_name", DEFAULT_WINDOW)),
        "target_fps": int(p.get("target_fps", DEFAULT_TARGET_FPS)),
        "db_floor": float(DB_FLOOR),
        "hide_low": ("" if hide_low is None else float(hide_low)),
        "colormap": ("" if colormap is None else str(colormap)),
        "duration_s": float(n_frames * hop) / float(sr),
        "source": ("" if source_path is None else os.path.basename(str(source_path))),
        "rows_are": "midi decreasing with row index; row 0 = MIDI_MAX",
    }


def export_result(path, payload, fmt=None):
    """把 payload 写到 path。fmt 为空时按扩展名推断。返回实际写入的路径。"""
    path = str(path)
    if fmt is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        fmt = {"npz": "npz", "npy": "npy", "csv": "csv",
               "bin": "raw(f32)", "raw": "raw(f32)", "f32": "raw(f32)"}.get(ext, "npz")
    mag = payload["mag"]
    db = payload["db"]
    meta = {k: v for k, v in payload.items()
            if k not in ("mag", "db", "u8") and not isinstance(v, np.ndarray)}

    if fmt == "npz":
        arrays = {"mag": mag, "db": db,
                  "time_s": payload["time_s"], "midi": payload["midi"],
                  "freq_hz": payload["freq_hz"]}
        if payload.get("u8") is not None:
            arrays["u8"] = payload["u8"]
        np.savez_compressed(path, **arrays, **meta)
        return path

    if fmt == "npy":
        np.save(path, mag)
        return path

    if fmt == "raw(f32)":
        mag.astype("<f4").tofile(path)
        return path

    if fmt == "csv":
        # 头部注释带元数据，之后第一列是时间，其余列按中音号命名
        lines = ["# wavetonepro export",
                 f"# shape(n_frames,n_rows)={db.shape[0]},{db.shape[1]}",
                 f"# rows_are={payload['rows_are']}"]
        for k in sorted(meta):
            lines.append(f"# {k}={meta[k]}")
        hdr = ["time_s"] + [f"midi{m:.3f}" for m in payload["midi"]]
        lines.append(",".join(hdr))
        t = payload["time_s"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
            block = max(1, 200000 // max(1, db.shape[1]))
            for i in range(0, db.shape[0], block):
                sub = db[i:i + block]
                for j in range(sub.shape[0]):
                    f.write(f"{t[i + j]:.6f}," +
                            ",".join(f"{v:.3f}" for v in sub[j]) + "\n")
        return path

    raise ValueError(f"未知导出格式: {fmt}")



# =========================================================================
# 分析工作线程
# =========================================================================
class AnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(object, int, object)      # (db, hop, mag)
    failed = pyqtSignal(str)

    def __init__(self, samples, sr, params, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.sr = sr
        self.params = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:

            def progress_cb(frac):
                pct = int(max(0.0, min(1.0, float(frac))) * 100)
                self.progress.emit(pct, f"分析中… {pct}%")

            def cancel_cb():
                return self._cancel

            mag, hop = compute_reassigned_spectrogram(
                self.samples,
                self.sr,
                midi_min=MIDI_MIN,
                midi_max=MIDI_MAX,
                rows_per_semitone=self.params["rows_per_semitone"],
                target_fps=self.params["target_fps"],
                window_name=self.params["window_name"],
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
        except AnalysisCancelled:
            self.failed.emit("__cancelled__")
            return
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.failed.emit(str(e))
            return

        if self._cancel:
            self.failed.emit("__cancelled__")
            return

        self.progress.emit(99, "计算 dB 中…")
        try:
            db = compute_db(mag)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.failed.emit(str(e))
            return

        self.progress.emit(100, "完成")
        # 同时带回未归一化的线性幅度，供导出使用（db 只用于显示）
        self.done.emit(db, hop, mag)


# =========================================================================
# 参数对话框
# =========================================================================
class AnalysisParamsDialog(QDialog):
    def __init__(self, params=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("分析参数")
        self.setModal(True)
        self.setMinimumWidth(320)

        params = params or {}

        layout = QFormLayout(self)

        self.cb_window = QComboBox()
        for name in WINDOW_CHOICES:
            self.cb_window.addItem(name)
        cur_w = params.get("window_name", DEFAULT_WINDOW)
        if cur_w in WINDOW_CHOICES:
            self.cb_window.setCurrentIndex(WINDOW_CHOICES.index(cur_w))
        self.cb_window.setToolTip("Hann 通用；BH 旁瓣更低；flattop 幅度最准")
        layout.addRow("窗函数", self.cb_window)

        self.cb_rows = QComboBox()
        for r in ROWS_PER_SEMITONE_CHOICES:
            self.cb_rows.addItem(str(r), r)
        cur_r = params.get("rows_per_semitone", DEFAULT_ROWS_PER_SEMITONE)
        try:
            self.cb_rows.setCurrentIndex(ROWS_PER_SEMITONE_CHOICES.index(cur_r))
        except ValueError:
            self.cb_rows.setCurrentIndex(1)
        self.cb_rows.setToolTip("频谱图纵轴分辨率（每半音的行数）")
        layout.addRow("每半音行数", self.cb_rows)

        self.cb_fps = QComboBox()
        for f in FPS_CHOICES:
            self.cb_fps.addItem(str(f), f)
        cur_f = params.get("target_fps", DEFAULT_TARGET_FPS)
        try:
            self.cb_fps.setCurrentIndex(FPS_CHOICES.index(cur_f))
        except ValueError:
            self.cb_fps.setCurrentIndex(2)
        self.cb_fps.setToolTip("时间分辨率（每秒帧数）")
        layout.addRow("时间分辨率 (fps)", self.cb_fps)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_params(self):
        return {
            "window_name": self.cb_window.currentText(),
            "rows_per_semitone": int(self.cb_rows.currentData()),
            "target_fps": int(self.cb_fps.currentData()),
        }


# =========================================================================
# 只在松手时汇报的滑块
# =========================================================================
class ReleaseSlider(QSlider):
    """拖动过程中不发信号，松手（或键盘/滚轮调整结束）才发一次。

    频谱图这类重算很贵，边拖边算是拖不动的根源。用这个滑块可以把
    "跟手的数值反馈"和"昂贵的重算"分开：数值标签自己实时更新，
    真正的处理挂在 valueReleased 上。
    """

    valueReleased = pyqtSignal(int)

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._dragging = False
        self.sliderReleased.connect(self._emit_released)

    def mousePressEvent(self, e):
        self._dragging = True
        super().mousePressEvent(e)

    def _emit_released(self):
        if not self._dragging:
            return          # 没有按下过的 release 事件（Qt 有时会补发）
        self._dragging = False
        self.valueReleased.emit(self.value())

    def wheelEvent(self, e):
        super().wheelEvent(e)
        self.valueReleased.emit(self.value())

    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        if e.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
                       Qt.Key_PageUp, Qt.Key_PageDown, Qt.Key_Home, Qt.Key_End):
            self.valueReleased.emit(self.value())

    def setValueSilently(self, v):
        """程序赋值：不触发任何处理，也不触发 valueReleased。"""
        blocked = self.blockSignals(True)
        self.setValue(int(v))
        self.blockSignals(blocked)


# =========================================================================
# 阶梯滤镜参数对话框（带实时预览）
# =========================================================================
class _SliderRow:
    """一行「滑块 + 数值」，把滑块整数线性映射到一个浮点区间。"""

    def __init__(self, parent_layout, label, lo, hi, default, fmt="{:.3f}",
                 tooltip="", on_change=None, on_release=None):
        self.lo, self.hi = float(lo), float(hi)
        self.fmt = fmt
        # 用 ReleaseSlider：拖动时只更新数字标签，松手才触发重算。
        # 重算一次要两百毫秒左右，边拖边算必然拖不动。
        self.slider = ReleaseSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(20)
        self.slider.setValue(self._to_slider(default))
        self.value_label = QLabel(self.fmt.format(default))
        self.value_label.setMinimumWidth(56)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet(
            "color:#9fc5ff;font-family:Consolas,Menlo,monospace;")
        if tooltip:
            self.slider.setToolTip(tooltip)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label, 0)
        parent_layout.addRow(label, row)
        self._on_change = on_change
        self._on_release = on_release
        self.slider.valueChanged.connect(self._on_slide)
        self.slider.valueReleased.connect(self._on_released)

    # --- 映射 ---
    def _to_slider(self, v):
        if self.hi <= self.lo:
            return 0
        return int(round((float(v) - self.lo) / (self.hi - self.lo) * 1000))

    def value(self):
        return self.lo + (self.hi - self.lo) * (self.slider.value() / 1000.0)

    def set_value(self, v):
        self.slider.setValueSilently(self._to_slider(v))

    def _on_slide(self, _v):
        """拖动中：只更新数字，不做任何重算。"""
        self.value_label.setText(self.fmt.format(self.value()))
        if self._on_change is not None:
            self._on_change()

    def _on_released(self, _v):
        """松手：数值已定，交给外层真正处理。"""
        self.value_label.setText(self.fmt.format(self.value()))
        if self._on_release is not None:
            self._on_release()
        elif self._on_change is not None:
            self._on_change()

    def setEnabled(self, on):
        self.slider.setEnabled(on)
        self.value_label.setEnabled(on)

    def refresh_label(self):
        self.value_label.setText(self.fmt.format(self.value()))


class FilterConfigDialog(QDialog):
    """配置阶梯滤镜：左侧调参，频谱图实时刷新。取消则恢复打开前的设置。

    滑块的语义由 ReleaseSlider 提供：拖动中只更新数字标签，
    松手后才真正重算。
    """

    REDUCE_LABELS = (
        ("半音内取平均", "mean"),
        ("去掉上下各 N 行后取最大", "midmax"),
    )
    CURVE_LABELS = (
        ("不压", "none"),
        ("knee：低于阈值整体下移", "knee"),
        ("幂曲线：以支点连续压缩", "power"),
        ("S 形：拐点以下压向底", "sigmoid"),
    )
    PREVIEW_DEBOUNCE_MS = 120

    def __init__(self, config, on_change=None, original=None, parent=None,
                 rows_per_semitone=DEFAULT_ROWS_PER_SEMITONE,
                 on_trim_user=None):
        super().__init__(parent)
        self.setWindowTitle("阶梯滤镜参数")
        self.setModal(False)
        self.setMinimumWidth(480)
        self._on_change = on_change
        self._on_trim_user = on_trim_user
        self._original = dict(original) if original else None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(self.PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._flush_preview)
        c = _filter_config(config)

        root = QFormLayout(self)

        # ---- 归约方式 ----
        self.cb_reduce = QComboBox()
        for label, val in self.REDUCE_LABELS:
            self.cb_reduce.addItem(label, val)
        vals = [v for _, v in self.REDUCE_LABELS]
        self.cb_reduce.setCurrentIndex(vals.index(c["reduce"]) if c["reduce"] in vals else 1)
        self.cb_reduce.setToolTip(
            f"每个半音压成一个特征值，作为这条半音的台阶高度。\n"
            f"「取平均」保留均值但抹掉亮点；「去掉上下各 N 行后取最大」\n"
            f"能在保住亮点的同时丢掉边界泄漏，台阶更干净。\n"
            f"当前每半音 {max(1, int(rows_per_semitone))} 行"
            f"{'——只有 1 行时这招没有意义，已禁用' if int(rows_per_semitone) <= 1 else ''}。")
        self.cb_reduce.currentIndexChanged.connect(self._apply_now)
        root.addRow("归约方式", self.cb_reduce)

        # ---- N（整数，用步进框更直接）----
        # 上限跟着 rows_per_semitone 走：一个半音只有 rps 行，去掉太多就没得取了
        self._rps = max(1, int(rows_per_semitone))
        self._trim_max = max(0, (self._rps - 1) // 2)
        self.sp_trim = QSpinBox()
        self.sp_trim.setRange(0, self._trim_max)
        self.sp_trim.setValue(scaled_trim(c["trim"], self._rps))
        self.sp_trim.setToolTip(
            f"去掉半音最上/最下的 N 行再取最大（0 = 等同整段取最大）。\n"
            f"当前每半音 {self._rps} 行，N 最大 {self._trim_max}，"
            f"去掉太多就没行可取了。\n"
            f"峰值通常在中间、泄漏堆在边界，去掉边界收益最大。\n"
            f"预设按 12 行去 3 行的比例折算：6 行→1，24 行→6，48 行→12。")
        # N 是连续可调的（按住方向键会连发），走防抖；下拉框是点一下就定了，立即生效
        self.sp_trim.valueChanged.connect(self._on_trim_changed)
        root.addRow("midmax 去掉 N 行", self.sp_trim)

        # ---- 曲线 ----
        self.cb_curve = QComboBox()
        for label, val in self.CURVE_LABELS:
            self.cb_curve.addItem(label, val)
        cvals = [v for _, v in self.CURVE_LABELS]
        self.cb_curve.setCurrentIndex(cvals.index(c["curve"]) if c["curve"] in cvals else 1)
        self.cb_curve.setToolTip(
            "归约之后再过一条曲线：弱半音继续压低、强半音基本不动，台阶更陡。\n"
            "所有曲线都只压不强推，避免过曝。")
        self.cb_curve.currentIndexChanged.connect(self._apply_now)
        root.addRow("曲线", self.cb_curve)

        # ---- 各曲线的滑块 ----
        # on_change 只负责刷新数字；真正重算挂在松手上（ReleaseSlider）。
        # 说明里的数值都是线性幅度（0..1），实现按 dB 折算：
        # 幅度 a 对应 20·lg(a) dB。
        self.s_knee_th = _SliderRow(
            root, "knee 阈值", 0.0, 1.0, c["knee_th"], "{:.3f}",
            "线性幅度阈值（0..1）。低于它的值整体乘以「knee 增益」。\n"
            "例如 0.15 约等于 -16.5 dB。", on_change=self._enabled_only,
            on_release=self._changed)
        self.s_knee_g = _SliderRow(
            root, "knee 增益", 0.0, 1.0, c["knee_gain"], "{:.2f}",
            "阈值以下乘以这个增益。0.5 约等于再降 6 dB，0 = 直接抹平。\n"
            "整段是同一个倍数，所以不会越暗被压得越狠。",
            on_change=self._enabled_only, on_release=self._changed)

        self.s_pow_g = _SliderRow(
            root, "幂 gamma", 1.0, 6.0, c["pow_gamma"], "{:.2f}",
            "幂压缩的强度。1 = 不压；越大压得越狠（暗部被推得更低）。\n"
            "与 knee 的区别：knee 在阈值处是折线，幂曲线是连续压缩。",
            on_change=self._enabled_only, on_release=self._changed)
        self.s_pow_p = _SliderRow(
            root, "幂支点", 0.01, 1.0, c["pow_pivot"], "{:.3f}",
            "线性幅度支点（0..1）。等于它的值不变，低于它按 gamma 压缩，\n"
            "高于它保持原值（不会提亮，避免过曝）。0.3 约等于 -10.5 dB。",
            on_change=self._enabled_only, on_release=self._changed)

        self.s_sig_c = _SliderRow(
            root, "S 拐点", 0.0, 1.0, c["sig_center"], "{:.3f}",
            "软限幅的拐点，按显示范围的比例给：0 = 峰值(0 dB)，\n"
            "1 = 显示下限(-100 dB)，所以 0.3 表示 -30 dB。\n"
            "拐点处压掉一半，往暗处逐渐压到底、往亮处逐渐不压。\n"
            "0 表示不压（等同关闭曲线）。",
            on_change=self._enabled_only, on_release=self._changed)
        self.s_sig_k = _SliderRow(
            root, "S 过渡宽度", 0.02, 1.0, c["sig_k"], "{:.2f}",
            "拐点上下各留多宽做平滑过渡，按拐点深度的比例给。\n"
            "越大过渡越平缓（接近线性压暗），越小越接近硬阈值。\n"
            "它和 knee 的差别就在这里：knee 是硬折线，这里是平滑过弯。",
            on_change=self._enabled_only, on_release=self._changed)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_reset = QPushButton("恢复默认")
        btns.addButton(self.btn_reset, QDialogButtonBox.ResetRole)
        self.btn_reset.clicked.connect(self._on_reset)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addRow(btns)

        self._update_enabled()
        self.resize(510, self.sizeHint().height())

    def get_config(self):
        return _filter_config({
            "reduce": self.cb_reduce.currentData(),
            "trim": int(self.sp_trim.value()),
            "curve": self.cb_curve.currentData(),
            "knee_th": self.s_knee_th.value(),
            "knee_gain": self.s_knee_g.value(),
            "pow_gamma": self.s_pow_g.value(),
            "pow_pivot": self.s_pow_p.value(),
            "sig_center": self.s_sig_c.value(),
            "sig_k": self.s_sig_k.value(),
        })

    def _update_enabled(self):
        """只让当前生效的参数可编辑，避免误调。"""
        # 每半音只有 1 行时 midmax 无意义（去任何行都会把唯一一行去掉）
        mid_ok = self._trim_max > 0
        item = self.cb_reduce.model().item(
            [v for _, v in self.REDUCE_LABELS].index("midmax"))
        if item is not None and item.isEnabled() != mid_ok:
            item.setEnabled(mid_ok)
        if not mid_ok and self.cb_reduce.currentData() == "midmax":
            self.cb_reduce.setCurrentIndex(
                [v for _, v in self.REDUCE_LABELS].index("mean"))
        self.sp_trim.setEnabled(mid_ok and self.cb_reduce.currentData() == "midmax")
        cur = self.cb_curve.currentData()
        self.s_knee_th.setEnabled(cur == "knee")
        self.s_knee_g.setEnabled(cur == "knee")
        self.s_pow_g.setEnabled(cur == "power")
        self.s_pow_p.setEnabled(cur == "power")
        self.s_sig_c.setEnabled(cur == "sigmoid")
        self.s_sig_k.setEnabled(cur == "sigmoid")

    def _on_trim_changed(self, v):
        """N 被改动：记下"用户手动设过"，之后换行数时不再自动折算。"""
        if not self._loading and self._on_trim_user is not None:
            self._on_trim_user(int(v))
        self._changed()

    def _enabled_only(self, *a):
        """滑块拖动中：只同步一下可用状态，不做重算。"""
        self._update_enabled()

    def _changed(self, *a):
        """滑块松手 / 其它控件变化：真正重算并刷新预览。"""
        self._update_enabled()
        self._preview_timer.start()

    def _flush_preview(self):
        self._preview_timer.stop()
        if self._on_change is not None:
            self._on_change(self.get_config())

    def _apply_now(self, *a):
        """下拉框这类"点一下就定了"的控件，立刻出图不用等。"""
        self._update_enabled()
        self._flush_preview()

    def _on_reset(self):
        c = _filter_config(None)
        self.cb_reduce.setCurrentIndex(
            [v for _, v in self.REDUCE_LABELS].index(c["reduce"]))
        self.cb_curve.setCurrentIndex(
            [v for _, v in self.CURVE_LABELS].index(c["curve"]))
        self.sp_trim.setValue(min(int(c["trim"]), self._trim_max))
        self.s_knee_th.set_value(c["knee_th"])
        self.s_knee_g.set_value(c["knee_gain"])
        self.s_pow_g.set_value(c["pow_gamma"])
        self.s_pow_p.set_value(c["pow_pivot"])
        self.s_sig_c.set_value(c["sig_center"])
        self.s_sig_k.set_value(c["sig_k"])
        for s in (self.s_knee_th, self.s_knee_g, self.s_pow_g, self.s_pow_p,
                  self.s_sig_c, self.s_sig_k):
            s.refresh_label()
        self._apply_now()

    def _finish(self, ok):
        """接受/取消前把挂起的预览落定，避免最后一次拖动被丢掉。"""
        self._preview_timer.stop()
        if ok:
            self._flush_preview()
        elif self._on_change is not None and self._original is not None:
            self._on_change(dict(self._original))

    def accept(self):
        self._finish(True)
        super().accept()

    def reject(self):
        self._finish(False)
        super().reject()


class ChordKeyboard(QWidget):
    """横向钢琴键盘，用来点选和弦音（两个八度）。

    布局与侧边栏钢琴一致（白键先铺满、黑键叠在上面），只是转了 90°：
    音高沿 x 增加，白键占满高度、黑键占上方 62% 高度。
    左键加入 / 右键移除，命中判定用清晰的"先黑键后白键"，不会误选。
    """

    notesChanged = pyqtSignal(list)

    BLACK_RATIO = 0.40      # 黑键宽度 / 白键宽度
    BLACK_H = 0.62          # 黑键高度 / 整高

    def __init__(self, midi_min=CUSTOM_CHORD_LO, midi_max=CUSTOM_CHORD_HI, parent=None):
        super().__init__(parent)
        self.midi_min = int(midi_min)
        self.midi_max = int(midi_max)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._sel = set()
        self._hover = None
        self._c_white = QColor(234, 238, 245)
        self._c_black = QColor(26, 30, 38)
        self._c_sel_w = QColor(255, 214, 120)
        self._c_sel_b = QColor(206, 156, 30)
        self._c_hover_w = QColor(196, 218, 255)
        self._c_hover_b = QColor(66, 92, 148)
        self._c_line = QColor(120, 128, 144)
        self._c_border = QColor(8, 10, 14)

    # ---------------- 几何 ----------------
    def _white_notes(self):
        return [m for m in range(self.midi_min, self.midi_max + 1)
                if (m % 12) not in BLACK_PC]

    def _white_w(self):
        n = max(1, len(self._white_notes()))
        return max(1.0, self.width() / float(n))

    def _white_index(self, m):
        """m 是第几个白键（黑键返回它左边那个白键的序号）。"""
        idx = 0
        for x in range(self.midi_min, m):
            if (x % 12) not in BLACK_PC:
                idx += 1
        return idx

    def white_rect(self, m):
        w = self._white_w()
        i = self._white_index(m)
        return QRectF(i * w, 0.0, w, float(self.height()))

    def black_rect(self, m):
        """黑键对称压在它左下白键与右下白键的边界上。

        宽度取白键的 0.4 并在边界两侧各 0.2：这样黑键不会盖住左右白键的中心，
        命中判定（先黑后白）对任何键的中心点都能选回自己。
        """
        left = m - 1
        while left >= self.midi_min and (left % 12) in BLACK_PC:
            left -= 1
        b = self.white_rect(left)
        w = b.width() * self.BLACK_RATIO
        boundary = b.right()
        return QRectF(boundary - w * 0.5, 0.0, w, self.height() * self.BLACK_H)

    def note_at(self, pos):
        """先判黑键（叠在上层），再判白键。"""
        for m in range(self.midi_min, self.midi_max + 1):
            if (m % 12) in BLACK_PC and self.black_rect(m).contains(pos.x(), pos.y()):
                return m
        for m in self._white_notes():
            if self.white_rect(m).contains(pos.x(), pos.y()):
                return m
        return None

    # ---------------- 选择 ----------------
    def selected(self):
        return sorted(self._sel)

    def set_selected(self, notes):
        """程序设置选中音。会发 notesChanged，让外层跟着更新。"""
        new = {int(n) for n in notes
               if self.midi_min <= int(n) <= self.midi_max}
        if new != self._sel:
            self._sel = new
            self.update()
            self.notesChanged.emit(self.selected())

    def _toggle(self, m, add):
        if m is None:
            return
        changed = False
        if add and m not in self._sel:
            self._sel.add(m)
            changed = True
        elif (not add) and m in self._sel:
            self._sel.discard(m)
            changed = True
        if changed:
            self.update()
            self.notesChanged.emit(self.selected())

    # ---------------- 事件 ----------------
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._toggle(self.note_at(e.pos()), True)
        elif e.button() == Qt.RightButton:
            self._toggle(self.note_at(e.pos()), False)

    def mouseMoveEvent(self, e):
        m = self.note_at(e.pos())
        if m != self._hover:
            self._hover = m
            self.update()

    def leaveEvent(self, e):
        if self._hover is not None:
            self._hover = None
            self.update()

    def wheelEvent(self, e):
        e.ignore()

    def contextMenuEvent(self, e):
        e.accept()          # 右键只用来移除，不弹菜单

    def paintEvent(self, ev):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(16, 19, 26))

        def pick(m, white_mode):
            if m in self._sel:
                return self._c_sel_w if white_mode else self._c_sel_b
            if m == self._hover:
                return self._c_hover_w if white_mode else self._c_hover_b
            return self._c_white if white_mode else self._c_black

        whites = self._white_notes()
        for i, m in enumerate(whites):
            r = self.white_rect(m)
            p.fillRect(r, pick(m, True))
        # 白键之间的分隔线
        p.setPen(QPen(self._c_border, 1))
        for i in range(1, len(whites)):
            x = i * self._white_w()
            p.drawLine(QPointF(x, 0.0), QPointF(x, float(H)))

        for m in range(self.midi_min, self.midi_max + 1):
            if (m % 12) not in BLACK_PC:
                continue
            r = self.black_rect(m)
            p.fillRect(r, pick(m, False))
            p.setPen(QPen(self._c_border, 1))
            p.drawRect(r.adjusted(0.0, 0.0, -0.5, -0.5))

        # C 的音名标注
        f = QFont()
        f.setPointSizeF(7.5)
        p.setFont(f)
        p.setPen(QColor(112, 122, 142))
        for m in whites:
            if m % 12 != 0:
                continue
            r = self.white_rect(m)
            if r.width() < 14:
                continue
            p.drawText(r, Qt.AlignBottom | Qt.AlignHCenter,
                       f"C{m // 12 - 1}")

        p.setPen(QPen(QColor(48, 55, 70), 1))
        p.drawRect(0, 0, W - 1, H - 1)


class ChordBuilderDialog(QDialog):
    """自定义和弦窗口：两个八度的横向钢琴，左键加入 / 右键移除。

    与主界面共用同一套和弦表示法（从低音往上数的半音间隔），
    所以在这里点出来的形状，就是鼠标在频谱上以任意低音画出来的形状。
    返回的间隔以**最低选中音**为 0 归一化。
    """

    def __init__(self, selected_notes, root_hint=CUSTOM_CHORD_LO,
                 on_change=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义和弦")
        self.setModal(False)
        self.setMinimumWidth(560)
        self._on_change = on_change

        root = QVBoxLayout(self)
        self.kb = ChordKeyboard(CUSTOM_CHORD_LO, CUSTOM_CHORD_HI, self)
        self.kb.set_selected(selected_notes)
        root.addWidget(self.kb, 1)

        self.lbl = QLabel("")
        self.lbl.setStyleSheet("color:#9fc5ff;font-family:Consolas,Menlo,monospace;")
        root.addWidget(self.lbl)

        hint = QLabel("左键点击加入和弦音 · 右键点击移除 · 关闭窗口后生效")
        hint.setStyleSheet("color:#8b96ad;")
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_clear = QPushButton("清空")
        btns.addButton(self.btn_clear, QDialogButtonBox.ResetRole)
        self.btn_clear.clicked.connect(lambda: self.kb.set_selected([]))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.kb.notesChanged.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        notes = self.kb.selected()
        if notes:
            base = notes[0]
            names = " ".join(midi_name(n) for n in notes)
            iv = [n - base for n in notes]
            self.lbl.setText(f"{names}   间隔 {iv}（相对最低音）")
        else:
            self.lbl.setText("未选择任何音")
        if self._on_change is not None:
            self._on_change(notes)

    def intervals(self):
        """归一化成"从低音往上数的半音间隔"。空选返回空元组（= 关闭和弦）。"""
        notes = self.kb.selected()
        if not notes:
            return ()
        base = notes[0]
        return tuple(n - base for n in notes)


# =========================================================================
# 谱面设置对话框
# =========================================================================
class SpectrumSettingsDialog(QDialog):
    """把显示相关的设置集中到一个窗口：音域 / 外观 / 小节线。

    这些选项都是"选一下就定了"，不需要防抖，改动立刻反映到频谱图上；
    取消则整体回滚到打开前的状态。
    """

    def __init__(self, settings, on_change=None, original=None,
                 has_data=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("谱面设置")
        self.setModal(False)
        self.setMinimumWidth(480)
        self._on_change = on_change
        self._original = dict(original) if original else None
        s = dict(settings)
        self._loading = True

        root = QFormLayout(self)

        # ---------------- 音域 ----------------
        # 用滑块而不是数字框：音域只有 21..45 与 84..108 两段有意义，
        # 而且 21+63=84，所以两边都拉到头刚好是完整键盘（中间不会出现空档）。
        grp_key = QGroupBox("钢琴窗显示音域")
        fk = QFormLayout(grp_key)
        self.sld_key_lo = ReleaseSlider(Qt.Horizontal, self)
        self.sld_key_lo.setRange(MIDI_MIN, KEY_SPAN_LO_MAX)
        self.sld_key_lo.setToolTip(
            f"最低显示音 {midi_name(MIDI_MIN)}–{midi_name(KEY_SPAN_LO_MAX)}\n"
            "往右调 = 隐藏底部半音，中间区域更宽")
        row_lo = QHBoxLayout()
        row_lo.setContentsMargins(0, 0, 0, 0)
        row_lo.addWidget(self.sld_key_lo, 1)
        self.lbl_key_lo = QLabel("")
        self.lbl_key_lo.setMinimumWidth(60)
        self.lbl_key_lo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_key_lo.setStyleSheet(
            "color:#9fc5ff;font-family:Consolas,Menlo,monospace;")
        row_lo.addWidget(self.lbl_key_lo)
        fk.addRow("最低音", row_lo)

        self.sld_key_hi = ReleaseSlider(Qt.Horizontal, self)
        self.sld_key_hi.setRange(KEY_SPAN_HI_MIN, MIDI_MAX)
        self.sld_key_hi.setToolTip(
            f"最高显示音 {midi_name(KEY_SPAN_HI_MIN)}–{midi_name(MIDI_MAX)}\n"
            "往左调 = 隐藏顶部半音，中间区域更宽")
        row_hi = QHBoxLayout()
        row_hi.setContentsMargins(0, 0, 0, 0)
        row_hi.addWidget(self.sld_key_hi, 1)
        self.lbl_key_hi = QLabel("")
        self.lbl_key_hi.setMinimumWidth(60)
        self.lbl_key_hi.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_key_hi.setStyleSheet(
            "color:#9fc5ff;font-family:Consolas,Menlo,monospace;")
        row_hi.addWidget(self.lbl_key_hi)
        fk.addRow("最高音", row_hi)

        self.btn_key_full = QPushButton(f"恢复全键盘 ({midi_name(MIDI_MIN)}–{midi_name(MIDI_MAX)})")
        self.btn_key_full.clicked.connect(self._reset_key_range)
        fk.addRow(self.btn_key_full)
        root.addRow(grp_key)

        # ---------------- 外观 ----------------
        grp_look = QGroupBox("外观")
        fl = QFormLayout(grp_look)
        self.cb_cmap = QComboBox()
        for name in ("magma", "inferno", "viridis", "ice"):
            self.cb_cmap.addItem(name, name)
        self.cb_cmap.setToolTip("频谱图的颜色映射")
        fl.addRow("主题", self.cb_cmap)

        # 亮度滤镜滑块。**必须由对话框自己创建**：
        # 它天生以对话框为 parent，窗口关闭时不会被顺手删掉（deleteLater 只删
        # 对话框本身，子控件会被 parent 机制留下），主窗口可以放心继续持有它。
        # 反过来把工具栏的滑块 addWidget 进来则会被重设 parent 到对话框，
        # 窗口一关对象就没了 —— 那正是之前 RuntimeError 的原因。
        self.sld_hide = ReleaseSlider(Qt.Horizontal, self)
        self.sld_hide.setRange(0, 100)
        self.sld_hide.setToolTip("亮度滤镜：提高它会把较暗的部分压成背景色（松手生效）")
        row_hide = QHBoxLayout()
        row_hide.setContentsMargins(0, 0, 0, 0)
        row_hide.addWidget(self.sld_hide, 1)
        self.lbl_hide_local = QLabel("")
        self.lbl_hide_local.setMinimumWidth(38)
        self.lbl_hide_local.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_hide_local.setStyleSheet(
            "color:#9fc5ff;font-family:Consolas,Menlo,monospace;")
        row_hide.addWidget(self.lbl_hide_local)
        fl.addRow("亮度滤镜", row_hide)
        root.addRow(grp_look)

        # ---------------- 小节线 ----------------
        grp_beat = QGroupBox("小节线 / 节拍线")
        fb = QFormLayout(grp_beat)
        self.chk_beats = QCheckBox("显示节拍线与小节线")
        self.chk_beats.toggled.connect(self._sync_enabled)
        fb.addRow(self.chk_beats)

        self.sp_bpm = QDoubleSpinBox()
        self.sp_bpm.setRange(20.0, 400.0)
        self.sp_bpm.setDecimals(2)
        self.sp_bpm.setSingleStep(1.0)
        self.sp_bpm.setToolTip("每分钟拍数，决定节拍线间距")
        fb.addRow("BPM", self.sp_bpm)

        self.sp_bpb = QSpinBox()
        self.sp_bpb.setRange(1, 16)
        self.sp_bpb.setToolTip("每小节拍数，决定小节线位置")
        fb.addRow("拍/小节", self.sp_bpb)
        root.addRow(grp_beat)

        # ---------------- 按钮 ----------------
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_reset = QPushButton("恢复默认")
        btns.addButton(self.btn_reset, QDialogButtonBox.ResetRole)
        self.btn_reset.clicked.connect(self._on_reset)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addRow(btns)

        # ---------------- 初值 ----------------
        self.sld_key_lo.setValueSilently(int(s.get("midi_min", MIDI_MIN)))
        self.sld_key_hi.setValueSilently(int(s.get("midi_max", MIDI_MAX)))
        idx = self.cb_cmap.findData(s.get("colormap", "magma"))
        self.cb_cmap.setCurrentIndex(max(0, idx))
        self.chk_beats.setChecked(bool(s.get("show_beats", True)))
        self.sp_bpm.setValue(float(s.get("bpm", 120.0)))
        self.sp_bpb.setValue(int(s.get("beats_per_bar", 4)))
        self.sld_hide.setValueSilently(int(s.get("volume_percent", 35)))
        if not has_data:
            grp_look.setEnabled(False)
            grp_beat.setEnabled(False)
        self._loading = False

        self.cb_cmap.currentIndexChanged.connect(self._apply)
        self.chk_beats.toggled.connect(self._apply)
        self.sp_bpm.valueChanged.connect(self._apply)
        self.sp_bpb.valueChanged.connect(self._apply)
        # 音域滑块：拖动中只更新音名标签，松手才真正重新裁剪
        for sl in (self.sld_key_lo, self.sld_key_hi):
            sl.valueChanged.connect(self._on_key_label)
            sl.valueReleased.connect(self._on_key_released)
        # 数值标签实时跟手，重算等松手（valueReleased）
        self.sld_hide.valueChanged.connect(self._on_volume_label)
        self.sld_hide.valueReleased.connect(self._on_volume_released)

        self._refresh_key_labels()
        self._on_volume_label(self.sld_hide.value())
        self._sync_enabled()
        self.resize(480, self.sizeHint().height())

    # ---------------- 音域 ----------------
    def _key_pair(self):
        return int(self.sld_key_lo.value()), int(self.sld_key_hi.value())

    def _on_key_label(self, _v=None):
        """拖动中：只刷新音名，不做任何裁剪。"""
        self._refresh_key_labels()

    def _on_key_released(self, _v=None):
        """松手：如果两边撞到一块了就推开，再应用。"""
        lo, hi = self._key_pair()
        if hi - lo < MIN_KEY_SPAN:
            if self.sender() is self.sld_key_lo:
                hi = min(MIDI_MAX, lo + MIN_KEY_SPAN)
            else:
                lo = max(MIDI_MIN, hi - MIN_KEY_SPAN)
            self.sld_key_lo.setValueSilently(lo)
            self.sld_key_hi.setValueSilently(hi)
        self._refresh_key_labels()
        self._apply()

    def _refresh_key_labels(self):
        lo, hi = self._key_pair()
        self.lbl_key_lo.setText(midi_name(lo))
        self.lbl_key_hi.setText(midi_name(hi))
        self.btn_key_full.setText(
            f"共 {hi - lo + 1} 个半音 · 恢复全键盘"
            f" ({midi_name(MIDI_MIN)}–{midi_name(MIDI_MAX)})")

    def _reset_key_range(self):
        self.sld_key_lo.setValueSilently(MIDI_MIN)
        self.sld_key_hi.setValueSilently(MIDI_MAX)
        self._refresh_key_labels()
        self._apply()

    # ---------------- 其它 ----------------
    def _sync_enabled(self):
        on = self.chk_beats.isChecked()
        self.sp_bpm.setEnabled(on)
        self.sp_bpb.setEnabled(on)

    def _on_volume_label(self, v):
        """拖动中只更新数字，不做任何重算。"""
        self.lbl_hide_local.setText(f"{v}%")

    def _on_volume_released(self, v):
        """松手后才真正应用。"""
        self._on_volume_label(v)
        self._apply()

    def get_settings(self):
        lo, hi = self._key_pair()
        out = {
            "midi_min": lo,
            "midi_max": hi,
            "colormap": self.cb_cmap.currentData(),
            "show_beats": bool(self.chk_beats.isChecked()),
            "bpm": float(self.sp_bpm.value()),
            "beats_per_bar": int(self.sp_bpb.value()),
            "volume_percent": int(self.sld_hide.value()),
        }
        return out

    def _apply(self, *a):
        if self._loading:
            return
        self._sync_enabled()
        if self._on_change is not None:
            self._on_change(self.get_settings())

    def _on_reset(self):
        self._loading = True
        self.sld_key_lo.setValueSilently(MIDI_MIN)
        self.sld_key_hi.setValueSilently(MIDI_MAX)
        self.cb_cmap.setCurrentIndex(max(0, self.cb_cmap.findData("magma")))
        self.chk_beats.setChecked(True)
        self.sp_bpm.setValue(120.0)
        self.sp_bpb.setValue(4)
        self.sld_hide.setValueSilently(35)
        self._on_volume_label(35)
        self._loading = False
        self._refresh_key_labels()
        self._apply()

    def _finish(self, ok):
        if ok:
            self._apply()
        elif self._on_change is not None and self._original is not None:
            self._on_change(dict(self._original))

    def accept(self):
        self._finish(True)
        super().accept()

    def reject(self):
        self._finish(False)
        super().reject()


# =========================================================================
# 钢琴卷帘
# =========================================================================
class PianoRoll(QWidget):
    notePressed = pyqtSignal(int)

    def __init__(self, midi_min=MIDI_MIN, midi_max=MIDI_MAX, parent=None):
        super().__init__(parent)
        self.midi_min = midi_min
        self.midi_max = midi_max
        self.setFixedWidth(96)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._hover = None
        self._pressed = None
        self._highlights = []
        self._c_white = QColor(234, 238, 245)
        self._c_black = QColor(26, 30, 38)
        self._c_pressed_w = QColor(110, 170, 255)
        self._c_pressed_b = QColor(40, 90, 200)
        self._c_hover_w = QColor(196, 218, 255)
        self._c_hover_b = QColor(66, 92, 148)
        self._c_hl_w = QColor(255, 235, 140)
        self._c_hl_b = QColor(190, 155, 30)
        self._c_line = QColor(160, 166, 178)
        self._c_border = QColor(8, 10, 14)

    def set_highlight(self, midi):
        midi = int(midi)
        if midi < 0:
            self.set_highlights([])
        else:
            self.set_highlights([midi])

    def set_highlights(self, midis):
        try:
            new = [int(m) for m in midis if int(m) >= 0]
        except Exception:
            new = []
        seen = set()
        uniq = []
        for n in new:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        if uniq != self._highlights:
            self._highlights = uniq
            self.update()

    def _key_rect(self, m):
        H = max(1, self.height())
        span = self.midi_max - self.midi_min + 1
        y_t = (self.midi_max + 0.5 - (m + 0.5)) / span * H
        y_b = (self.midi_max + 0.5 - (m - 0.5)) / span * H
        return QRectF(0.0, y_t, float(self.width()), y_b - y_t)

    def _midi_at_y(self, y):
        H = max(1, self.height())
        span = self.midi_max - self.midi_min + 1
        v = (self.midi_max + 0.5) - (y / H) * span
        return max(self.midi_min, min(self.midi_max, int(round(v))))

    def paintEvent(self, ev):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(16, 19, 26))
        highlight_set = set(self._highlights)

        def pick(m, white_mode):
            if m == self._pressed:
                return self._c_pressed_w if white_mode else self._c_pressed_b
            if m == self._hover:
                return self._c_hover_w if white_mode else self._c_hover_b
            if m in highlight_set:
                return self._c_hl_w if white_mode else self._c_hl_b
            return self._c_white if white_mode else self._c_black

        for m in range(self.midi_min, self.midi_max + 1):
            if (m % 12) in BLACK_PC:
                continue
            r = self._key_rect(m)
            p.fillRect(r, pick(m, True))
            if m % 12 in (0, 5):
                p.setPen(QPen(self._c_line, 1))
                p.drawLine(QPointF(0.0, r.bottom()), QPointF(r.right(), r.bottom()))

        for m in range(self.midi_min, self.midi_max + 1):
            if (m % 12) not in BLACK_PC:
                continue
            r = self._key_rect(m)
            full_w = r.width()
            black_w = full_w * 0.62
            right_x = r.x() + black_w
            right_w = full_w - black_w
            mid_y = r.y() + r.height() * 0.5
            top_half = QRectF(right_x, r.y(), right_w, r.height() * 0.5)
            bottom_half = QRectF(right_x, mid_y, right_w, r.height() * 0.5)
            if m + 1 <= self.midi_max:
                p.fillRect(top_half, pick(m + 1, True))
            if m - 1 >= self.midi_min:
                p.fillRect(bottom_half, pick(m - 1, True))
            p.setPen(QPen(self._c_line, 1))
            p.drawLine(QPointF(right_x, mid_y), QPointF(r.right(), mid_y))
            left_rect = QRectF(r.x(), r.y(), black_w, r.height())
            p.fillRect(left_rect, pick(m, False))
            p.setPen(QPen(self._c_border, 1))
            p.drawRect(left_rect.adjusted(0.0, 0.0, -0.5, -0.5))

        f = QFont()
        f.setPointSizeF(7.2)
        p.setFont(f)
        p.setPen(QColor(112, 122, 142))
        for m in range(self.midi_min, self.midi_max + 1):
            if m % 12 != 0:
                continue
            r = self._key_rect(m)
            if r.height() < 8.5:
                continue
            octv = m // 12 - 1
            p.drawText(QRectF(W * 0.30, r.top(), W * 0.64, r.height()), Qt.AlignVCenter | Qt.AlignRight, f"C{octv}")

        p.setPen(QPen(QColor(48, 55, 70), 1))
        p.drawLine(W - 1, 0, W - 1, H)

    def mouseMoveEvent(self, e):
        m = self._midi_at_y(e.pos().y())
        if m != self._hover:
            self._hover = m
            self.update()

    def leaveEvent(self, e):
        if self._hover is not None:
            self._hover = None
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._pressed = self._midi_at_y(e.pos().y())
            self.notePressed.emit(self._pressed)
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._pressed = None
            self.update()

    def set_midi_range(self, midi_min, midi_max):
        """限制钢琴窗显示的音域，隐藏掉底部/顶部的半音。"""
        lo = max(0, min(int(midi_min), int(midi_max) - 1))
        hi = min(127, max(int(midi_max), lo + 2))
        if lo == self.midi_min and hi == self.midi_max:
            return
        self.midi_min, self.midi_max = lo, hi
        self._hover = None
        self._pressed = None
        self.update()

    def wheelEvent(self, e):
        e.ignore()


# =========================================================================
# 频谱图
# =========================================================================
class SpectrogramView(QWidget):
    hoverInfo = pyqtSignal(str)
    hoverNote = pyqtSignal(int)
    hoverNotes = pyqtSignal(list)
    clicked = pyqtSignal(float, int)
    noteTriggered = pyqtSignal(int)
    maskSelected = pyqtSignal(int)
    followModeChanged = pyqtSignal(bool)

    def __init__(self, midi_min=MIDI_MIN, midi_max=MIDI_MAX, parent=None):
        super().__init__(parent)
        self.midi_min = midi_min
        self.midi_max = midi_max
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 240)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)

        self.db = None
        self.u8 = None
        self.qimg = None
        self.n_frames = 0
        self.n_rows = 0
        self.hop = 512
        self.sr = 44100

        self.cmap = "magma"
        self.lut = LUTS[self.cmap]
        self.rgb_lut = LUTS_RGB[self.cmap]
        self.hide_low = 0.0
        self.show_grid = True

        # 阶梯滤镜：半音内取平均，只影响显示
        self.rows_per_semitone = DEFAULT_ROWS_PER_SEMITONE
        self.step_filter = False
        self.step_config = dict(DEFAULT_FILTER_CONFIG)
        self._step_cache = None
        self._trim_user_set = False  # 用户是否手动改过 midmax 的 N

        self.harmonics = 0
        # 和弦模式：intervals 为空表示关闭（普通单音）。转位只改形状，低音仍是鼠标位置。
        self.chord_intervals = ()
        self.chord_inversion = 0
        self.playhead_frame = None
        self.view_start = 0.0
        self.scale = 1.0

        self.masks = []
        self._mask_drag = None
        self.selected_mask = None

        self.follow_mode = False
        self._follow_ratio = 0.25
        self._has_manual_seek = False

        self.bpm = 120.0
        self.beats_per_bar = 4
        self.show_beats = True
        self.beat_offset_frames = 0.0

        self._cache = None
        self._drag_x0 = None
        self._drag_view0 = 0.0
        self._press_pos = None
        self._drag_moved = False
        self._hover = None

    def set_follow_mode(self, on):
        on = bool(on)
        if on == self.follow_mode:
            return
        self.follow_mode = on
        if on:
            if not self._has_manual_seek:
                self._follow_ratio = 0.25
            self._apply_follow()
        self._repaint()

    def reset_follow_state(self):
        self._follow_ratio = 0.25
        self._has_manual_seek = False

    def _cancel_follow(self):
        changed = False
        if self.follow_mode:
            self.follow_mode = False
            changed = True
        self._follow_ratio = 0.25
        self._has_manual_seek = False
        if changed:
            self.followModeChanged.emit(False)

    def _apply_follow(self):
        if self.n_frames <= 0 or self.playhead_frame is None:
            return
        W = max(1, self.width())
        visible = W / self.scale
        if visible >= self.n_frames:
            return
        new_view_start = self.playhead_frame - self._follow_ratio * visible
        max_view_start = self.n_frames - visible
        new_view_start = max(0.0, min(new_view_start, max_view_start))
        if abs(new_view_start - self.view_start) * self.scale > 0.5:
            self.view_start = new_view_start
            self._invalidate()

    def set_data(self, db, hop, sr, fit=True, rows_per_semitone=None):
        self.db = db
        self.hop = hop
        self.sr = sr
        self.n_frames, self.n_rows = db.shape
        if rows_per_semitone is not None and int(rows_per_semitone) != self.rows_per_semitone:
            old_rps = self.rows_per_semitone
            self.rows_per_semitone = int(rows_per_semitone)
            self._rescale_trim(old_rps)
        self._step_cache = None
        self._regen_u8()
        if self.selected_mask is not None:
            self.selected_mask = None
            self.maskSelected.emit(-1)
        self._has_manual_seek = False
        self._follow_ratio = 0.25
        if fit:
            QTimer.singleShot(0, self.fit_view)
        else:
            self._clamp_view()
            self._invalidate()

    def _rescale_trim(self, old_rps):
        """每半音行数变了，把 midmax 的 trim 按比例折算，保持视觉权重。

        12 行去 3 行 → 6 行去 1 行、24 行去 6 行、48 行去 12 行。
        用户在窗口里手动改过 N 就不动它（尊重手动值）。
        """
        old_rps = max(1, int(old_rps))
        if old_rps == self.rows_per_semitone:
            return
        if not self._trim_user_set:
            self.step_config = _filter_config(self.step_config)
            self.step_config["trim"] = scaled_trim(
                DEFAULT_FILTER_CONFIG["trim"], self.rows_per_semitone)
        else:
            self.step_config = _filter_config(self.step_config)
        self._step_cache = None

    def _set_trim_user(self, v):
        """用户在参数窗口里动过 N，之后不再自动折算。"""
        self._trim_user_set = True

    def set_colormap(self, name):
        if name not in LUTS:
            return
        self.cmap = name
        self.lut = LUTS[name]
        self.rgb_lut = LUTS_RGB[name]
        self._rebuild_qimg()
        self._invalidate_bg()

    def _rebuild_qimg(self):
        """按当前显示的音域裁出 QImage。

        只保留 [midi_min, midi_max] 对应的那些行，于是隐藏底部/顶部半音时
        中间区域会自动被拉高，其余绘制逻辑一行都不用改 —— 它们都是通过
        midi_to_y / y_to_midi 用 midi_min/midi_max 换算的。
        """
        if self.u8 is None:
            self.qimg = None
            return
        full = u8_to_qimage_fast(self.u8, self.rgb_lut)
        lo, hi = self._visible_row_span()
        if lo == 0 and hi == full.height():
            self.qimg = full
            return
        self.qimg = full.copy(0, lo, full.width(), max(1, hi - lo))

    def _qimg_frames(self):
        """当前 QImage 覆盖的帧数。

        粗预览只抽稀频率轴，帧数不变，所以正常就是 n_frames；
        万一将来改成抽帧，也按图自己的列数走，避免时间轴被显示错。
        """
        if self.qimg is not None and self.qimg.width() > 0:
            return self.qimg.width()
        return max(1, int(self.n_frames))

    def _visible_row_span(self):
        """当前音域占的行区间 [lo, hi)，一律基于**当前 u8 的实际高度**。

        阶梯滤镜开启时 u8 是"一个半音一行"的紧凑形式，此时一个半音正好是 1 行；
        否则一个半音占 rows_per_semitone 行。两种情况都按 u8 自己的高度夹住，
        所以不可能越界。
        """
        if self.u8 is None:
            return 0, max(1, self.n_rows)
        h = self.u8.shape[1]
        top = MIDI_MAX - int(self.midi_max)
        bottom = MIDI_MAX - int(self.midi_min) + 1
        if self.step_filter and self._compact_bands():
            lo = max(0, min(h, top))
            hi = max(lo + 1, min(h, bottom))
            return lo, hi
        rps = max(1, int(self.rows_per_semitone))
        lo = max(0, min(h, top * rps))
        hi = max(lo + 1, min(h, bottom * rps))
        return lo, hi

    def set_midi_range(self, midi_min, midi_max, rebuild=True):
        """设置可见音域（隐藏底部/顶部的若干半音）。"""
        lo = max(0, min(int(midi_min), int(midi_max) - 1))
        hi = min(127, max(int(midi_max), lo + 2))
        if lo == self.midi_min and hi == self.midi_max:
            return
        self.midi_min, self.midi_max = lo, hi
        if self.selected_mask is not None and self.selected_mask >= len(self.masks):
            self.selected_mask = None
            self.maskSelected.emit(-1)
        if rebuild and self.u8 is not None:
            self._regen_u8()      # 阶梯滤镜按新的音域重新归约
        self._clamp_view()
        self._invalidate_bg()

    def set_hide_low(self, v):
        v = float(v)
        if abs(v - self.hide_low) < 1e-6:
            return
        self.hide_low = v
        if self.db is not None:
            self._regen_u8()
        self._invalidate_bg()

    def set_step_filter(self, on):
        """半音内取平均的阶梯滤镜。只影响显示，不动分析结果。"""
        on = bool(on)
        if on == self.step_filter:
            return
        self.step_filter = on
        self._regen_u8()
        self._invalidate()

    def _display_db(self):
        """返回用于上色的 dB。

        阶梯滤镜开启时用缓存的滤波结果：能整除就返回"一个半音一行"的紧凑形式
        ((n_frames, n_semi))，否则退回逐行铺开的完整形式。
        """
        if not self.step_filter or self.db is None:
            return self.db
        if self._step_cache is None:
            compact = self._compact_bands()
            fn = semitone_bands if compact else semitone_step_filter
            try:
                self._step_cache = fn(self.db, self.rows_per_semitone,
                                      self.midi_max, config=self.step_config)
            except Exception as e:
                print(f"[filter] 阶梯滤镜失败: {type(e).__name__}: {e}")
                self._step_cache = self.db
        return self._step_cache

    def set_step_config(self, cfg):
        """更新阶梯滤镜参数并重算（预览用）。"""
        self.step_config = _filter_config(cfg)
        self._step_cache = None
        self._regen_u8()
        self._invalidate_bg()

    def set_harmonics(self, n):
        """改变高亮的泛音数量。

        遮罩是**按泛音数量画出来再烘进背景缓存**的（每个泛音一条色带），
        所以这里必须让背景缓存失效 —— 只重画覆盖层的话，遮罩会停在旧的
        泛音数量上，要等下一次创建/选中遮罩才刷新。
        """
        n = max(0, min(5, int(n)))
        if n != self.harmonics:
            self.harmonics = n
            if self._hover is not None and self.n_frames > 0:
                self._emit_hover_info(self._hover)
            if self.selected_mask is not None:
                if self.selected_mask >= len(self.masks):
                    self.selected_mask = None
                    self.maskSelected.emit(-1)
            self._invalidate_bg()

    def set_playhead_frame(self, frame):
        if frame is None:
            if self.playhead_frame is not None:
                self.playhead_frame = None
                self._repaint()
            return
        if self.playhead_frame is not None and abs(frame - self.playhead_frame) < 0.01:
            return
        self.playhead_frame = float(frame)
        if self.follow_mode and self.n_frames > 0:
            self._apply_follow()      # 视图真的移动时它会自己失效缓存
        self._repaint()

    def set_bpm(self, bpm):
        bpm = float(bpm)
        if bpm <= 0 or abs(bpm - self.bpm) < 1e-9:
            return
        self.bpm = bpm
        self._invalidate_bg()          # 节拍线画在缓存里

    def set_beats_per_bar(self, n):
        n = max(1, int(n))
        if n != self.beats_per_bar:
            self.beats_per_bar = n
            self._invalidate_bg()

    def set_show_beats(self, on):
        on = bool(on)
        if on != self.show_beats:
            self.show_beats = on
            self._invalidate_bg()

    def set_beat_offset_frames(self, frames):
        frames = float(frames)
        if abs(frames - self.beat_offset_frames) < 1e-9:
            return
        self.beat_offset_frames = frames
        self._invalidate_bg()

    def clear_masks(self):
        if self.masks or self._mask_drag is not None or self.selected_mask is not None:
            self.masks.clear()
            self._mask_drag = None
            self.selected_mask = None
            self.maskSelected.emit(-1)
            self._invalidate_bg()

    def _harmonic_notes(self, midi_n):
        """midi_n 的基音 + 泛音（供钢琴窗/状态栏高亮）。"""
        return self._harmonic_notes_of([midi_n])

    def _harmonic_notes_of(self, base_notes):
        """一组音的基音 + 各自泛音，合并去重后升序。

        用 sorted(set(...))：泛音可能落在别的基音上，也可能彼此重合，
        升序去重后钢琴窗与悬停高亮都不需要再关心顺序。
        """
        notes = []
        for m in base_notes:
            notes.append(int(round(m)))
            for k in range(2, self.harmonics + 2):
                mf = m + 12.0 * math.log2(k)
                if self.midi_min - 0.5 <= mf <= self.midi_max + 0.5:
                    notes.append(int(round(mf)))
        return sorted(set(notes))

    def _mask_at_pos(self, pos):
        if not self.masks:
            return None
        H = max(1, self.height())
        fx = self.view_start + pos.x() / self.scale
        py = pos.y()
        for idx, (f_start, f_end, midi_n) in enumerate(self.masks):
            if not (f_start <= fx <= f_end):
                continue
            for k in range(1, self.harmonics + 2):
                mf = midi_n + 12.0 * math.log2(k)
                y_top = midi_to_y(mf + 0.5, H, self.midi_min, self.midi_max)
                y_bot = midi_to_y(mf - 0.5, H, self.midi_min, self.midi_max)
                if y_top <= py <= y_bot:
                    return idx
        return None

    def _draw_mask_one(self, p, W, H, f_start, f_end, midi_n, fill):
        if f_end <= f_start:
            return
        x0 = (f_start - self.view_start) * self.scale
        x1 = (f_end - self.view_start) * self.scale
        x0c = max(0.0, x0)
        x1c = min(float(W), x1)
        if x1c <= x0c:
            return
        rect_w = x1c - x0c
        for k in range(1, self.harmonics + 2):
            mf = midi_n + 12.0 * math.log2(k)
            y_top = midi_to_y(mf + 0.5, H, self.midi_min, self.midi_max)
            y_bot = midi_to_y(mf - 0.5, H, self.midi_min, self.midi_max)
            y_a = max(0.0, min(float(H), y_top))
            y_b = max(0.0, min(float(H), y_bot))
            if y_b <= y_a:
                continue
            p.fillRect(QRectF(x0c, y_a, rect_w, y_b - y_a), fill)

    def _draw_masks(self, p, W, H):
        if not self.masks and self._mask_drag is None:
            return
        p.setPen(Qt.NoPen)
        fill_normal = QColor(0, 0, 0, 250)
        fill_selected = QColor(255, 255, 255, 250)
        for idx, (f_start, f_end, midi_n) in enumerate(self.masks):
            fill = fill_selected if idx == self.selected_mask else fill_normal
            self._draw_mask_one(p, W, H, f_start, f_end, midi_n, fill)
        if self._mask_drag is not None:
            f_start, f_end, midi_n = self._mask_drag
            self._draw_mask_one(p, W, H, f_start, f_end, midi_n, fill_normal)

    def _draw_beats(self, p, W, H):
        if (not self.show_beats) or self.n_frames <= 0 or self.sr <= 0 or self.hop <= 0:
            return
        if self.bpm <= 0 or self.scale <= 0:
            return
        frames_per_beat = (60.0 / float(self.bpm)) * self.sr / float(self.hop)
        if frames_per_beat <= 1e-6:
            return
        px_per_beat = frames_per_beat * self.scale
        px_per_bar = px_per_beat * self.beats_per_bar
        draw_beat_lines = px_per_beat >= 6.0
        draw_bar_lines = px_per_bar >= 6.0
        if not draw_beat_lines and not draw_bar_lines:
            return

        f0 = self.view_start
        f1 = self.view_start + W / self.scale
        i0 = max(0, int(math.floor((f0 - self.beat_offset_frames) / frames_per_beat)) - 1)
        i1 = int(math.ceil((f1 - self.beat_offset_frames) / frames_per_beat)) + 1
        if i1 - i0 > 20000:
            i1 = i0 + 20000

        beat_pen = QPen(QColor(130, 210, 255, 60), 1, Qt.DashLine)
        bar_pen = QPen(QColor(255, 190, 100, 165), 1, Qt.DashLine)

        for i in range(i0, i1 + 1):
            is_bar = (i % self.beats_per_bar) == 0
            if is_bar:
                if not draw_bar_lines:
                    continue
                p.setPen(bar_pen)
            else:
                if not draw_beat_lines:
                    continue
                p.setPen(beat_pen)
            frame = self.beat_offset_frames + i * frames_per_beat
            x = (frame - self.view_start) * self.scale
            if x < -1.0 or x > W + 1.0:
                continue
            p.drawLine(QPointF(x, 0.0), QPointF(x, float(H)))

    def _compact_bands(self):
        """阶梯滤镜开启且行数能整除时，直接按"一个半音一行"工作。

        这样上色和 QImage 构造的像素量都只有原来的 1/rows_per_semitone，
        绘制时由 QPainter 纵向拉伸补满（每个半音占的行数相同，位置不变）。
        """
        if not self.step_filter or self.db is None:
            return False
        rps = max(1, int(self.rows_per_semitone))
        return rps > 1 and self.n_rows > rps and self.n_rows % rps == 0

    def _regen_u8(self):
        """把当前 dB 上色成 u8 并生成 QImage。

        阶梯滤镜开启时先压成"一个半音一行"再上色，省掉 np.repeat 铺开和
        随之而来的 rps 倍重复上色 —— 这是滤镜路径最大的一笔开销。
        """
        if self.db is None:
            self.u8 = None
            self.qimg = None
            return
        src = self._display_db()
        self.u8 = db_to_u8(src, floor_db=DB_FLOOR,
                           hide_low=self.hide_low, gamma=DISPLAY_GAMMA)
        self._rebuild_qimg()

    def fit_view(self):
        W = max(1, self.width())
        if self.n_frames > 0:
            self.scale = W / float(self.n_frames)
            self.view_start = 0.0
        self._clamp_view()
        self._invalidate()

    def _min_scale(self):
        W = max(1, self.width())
        if self.n_frames <= 0:
            return 1e-5
        return W / float(self.n_frames)

    def _clamp_view(self):
        if self.n_frames <= 0:
            self.view_start = 0.0
            return
        W = max(1, self.width())
        ms = W / float(self.n_frames)
        if self.scale < ms:
            self.scale = ms
        visible = W / self.scale
        if visible >= self.n_frames:
            self.view_start = 0.0
            return
        max_start = self.n_frames - visible
        self.view_start = min(max(self.view_start, 0.0), max_start)

    # ------------------------------------------------------------------
    # 缓存失效的三档语义。改任何显示状态前先看这里，选错档就会出现
    # "改了设置但画面没跟着变"的 bug（遮罩的泛音数就踩过一次）。
    #
    #   _invalidate()     视图变换：缩放、平移、适应窗口、尺寸变化
    #   _invalidate_bg()  背景内容变了 —— 烘进缓存的东西：
    #                       qimg/u8        数据、色图、亮度、音域、阶梯滤镜
    #                       masks          遮罩增删改、选中态、**泛音数量**
    #                       midi_min/max   音域（同时影响网格线与遮罩位置）
    #                       show_grid      八度网格线
    #                       show_beats/bpm/beats_per_bar/beat_offset_frames
    #   _repaint()        只重画覆盖层，缓存不动 —— 只限这两项：
    #                       playhead_frame 播放头
    #                       _hover         悬停十字与信息框（泛音标记是覆盖层）
    # ------------------------------------------------------------------
    def _invalidate(self):
        """静态层需要重画（滚轮/缩放/平移等）。"""
        self._cache = None
        self.update()

    def _invalidate_bg(self):
        """背景缓存失效。凡是烘进缓存的状态改了都要走这里。"""
        self._cache = None
        self.update()

    def _repaint(self):
        """只重画覆盖层（播放头、悬停提示），背景缓存保持不动。

        跟随播放时每帧都会走到这里：省掉遮罩与节拍线的重绘是关键，
        它们原本是纯静态内容却每帧都在画。
        """
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        W = max(1, self.width())
        if self.n_frames > 0:
            ms = W / float(self.n_frames)
            if self.scale < ms:
                self.scale = ms
        self._clamp_view()
        if self.follow_mode:
            self._apply_follow()
        self._invalidate()

    def _render_cache(self):
        """把**画面里所有静态内容**一次性画进缓存位图。

        包含频谱图本身、遮罩、八度网格线、节拍线。这样每帧只需要
        drawPixmap + 播放头 + 悬停提示，而不是重画上面这一堆。
        """
        W, H = max(1, self.width()), max(1, self.height())
        if self._cache is None or self._cache.size() != QSize(W, H):
            self._cache = QPixmap(W, H)
        pm = self._cache
        pm.fill(QColor(7, 9, 14))
        p = QPainter(pm)
        try:
            if self.qimg is not None and self.n_frames > 0:
                # 时间轴按**这张图自己的列数**映射（阶梯滤镜紧凑模式下高度会
                # 变成半音数，宽度始终等于 n_frames，这里统一按图的宽度算）。
                q_frames = float(self._qimg_frames())
                f0 = self.view_start
                f1 = self.view_start + W / self.scale
                sx0 = max(0.0, f0)
                sx1 = min(q_frames, f1)
                if sx1 > sx0:
                    # 只在"放大"时开平滑：横向缩小时（把几万列压进一千多像素）
                    # 双线性既慢又没收益，直接关掉，跟随播放会明显更跟手。
                    shrink = (sx1 - sx0) / max(1.0, W)
                    p.setRenderHint(QPainter.SmoothPixmapTransform, shrink < 1.5)
                    dx0 = (sx0 - f0) * self.scale
                    dx1 = (sx1 - f0) * self.scale
                    # qimg 只含当前显示音域的那几行，源矩形用它自己的高度
                    rows = self.qimg.height()
                    src = QRectF(sx0, 0.0, sx1 - sx0, float(rows))
                    dst = QRectF(dx0, 0.0, dx1 - dx0, float(H))
                    p.drawImage(dst, self.qimg, src)
                    p.setRenderHint(QPainter.SmoothPixmapTransform, False)

            self._draw_masks(p, W, H)

            if self.show_grid:
                p.setPen(QPen(QColor(255, 255, 255, 26), 1))
                for m in range(self.midi_min, self.midi_max + 1):
                    if m % 12 != 0:
                        continue
                    y = midi_to_y(m, H, self.midi_min, self.midi_max)
                    if 0 <= y <= H:
                        p.drawLine(0, int(y), W, int(y))

            self._draw_beats(p, W, H)
        finally:
            p.end()

    # ---------------- 和弦 ----------------
    def set_chord(self, intervals, inversion=0):
        """设置和弦。intervals 为空即关闭和弦模式（回到普通单音）。

        间隔表按"从低音往上数"给出，转位只改变形状，低音始终是鼠标所指的音。
        """
        iv = tuple(sorted(set(int(x) for x in intervals)))
        if iv == self.chord_intervals and int(inversion) == self.chord_inversion:
            return
        self.chord_intervals = iv
        self.chord_inversion = int(inversion)
        if self._hover is not None and self.n_frames > 0:
            self._emit_hover_info(self._hover)
        # 悬停标记是覆盖层，但遮罩是按和弦音画进背景缓存的
        self._invalidate_bg()

    def chord_bass(self, midi_n):
        """鼠标所在音 = 和弦低音。非和弦模式就是它自己。"""
        n = int(round(midi_n))
        return max(MIDI_MIN, min(MIDI_MAX, n))

    def chord_notes(self, midi_n):
        """当前和弦在 midi_n 处要画的音（升序、去重、裁到可见音域内）。

        非和弦模式返回 [midi_n]。和弦音可能越过 MIDI_MAX，这里按显示音域裁掉。
        """
        bass = self.chord_bass(midi_n)
        if not self.chord_intervals:
            return [bass]
        out = []
        for n in chord_note_set(bass, self.chord_intervals, self.chord_inversion):
            if MIDI_MIN <= n <= MIDI_MAX and n not in out:
                out.append(n)
        return out or [bass]

    def _active_notes(self, midi_n):
        """用于悬停标记的音：和弦音（若是和弦）再加上它们的泛音。"""
        base = self.chord_notes(midi_n)
        if self.harmonics <= 0:
            return [(n, True) for n in base]
        offset = 12.0 * math.log2(2)     # 第 2 泛音 = 高八度
        marks = [(float(n), 1) for n in base]      # level 1 = 和弦音本身
        for n in base:
            for k in range(2, self.harmonics + 2):
                mf = n + 12.0 * math.log2(k)
                if self.midi_min - 1.0 <= mf <= self.midi_max + 1.0:
                    marks.append((mf, min(k, 3)))  # level 2/3 = 泛音层级
        return marks

    def _draw_harmonic_marks(self, p, midi_n, W, H):
        """画和弦音 + 泛音。

        层级决定配色：和弦音最亮（白），第 2/第 3 泛音用蓝色系区分，
        更高泛音再淡一档。非和弦模式下这就是原来的"基音 + 泛音"。
        """
        chord_on = bool(self.chord_intervals)
        for mf, level in self._active_notes(midi_n):
            y_top = midi_to_y(mf + 0.5, H, self.midi_min, self.midi_max)
            y_bot = midi_to_y(mf - 0.5, H, self.midi_min, self.midi_max)
            y_a = max(0.0, min(float(H), y_top))
            y_b = max(0.0, min(float(H), y_bot))
            if y_b <= y_a:
                continue
            if level == 1:
                # 和弦音 / 基音：白色；和弦音用暖白以便和泛音区分
                if chord_on:
                    fill_col = QColor(255, 226, 170, 120)
                    line_col = QColor(255, 226, 170, 225)
                else:
                    fill_col = QColor(255, 255, 255, 130)
                    line_col = QColor(255, 255, 255, 220)
            elif level == 2:
                fill_col = QColor(150, 210, 255, 95)
                line_col = QColor(150, 210, 255, 185)
            else:
                fill_col = QColor(130, 180, 235, 60)
                line_col = QColor(130, 180, 235, 130)
            p.fillRect(QRectF(0.0, y_a, float(W), y_b - y_a), fill_col)
            p.setPen(QPen(line_col, 1))
            if 0 <= y_top <= H:
                p.drawLine(QPointF(0.0, y_top + 0.5), QPointF(float(W), y_top + 0.5))
            if 0 <= y_bot <= H:
                p.drawLine(QPointF(0.0, y_bot - 0.5), QPointF(float(W), y_bot - 0.5))

    def _draw_playhead(self, p, W, H):
        if self.playhead_frame is None or self.n_frames <= 0:
            return
        x = (self.playhead_frame - self.view_start) * self.scale
        if -5 <= x <= W + 5:
            x = max(0.0, min(float(W), x))
            p.setPen(QPen(QColor(255, 70, 70, 230), 2))
            p.drawLine(QPointF(x, 0.0), QPointF(x, float(H)))
            tri = QPolygonF([QPointF(x - 6, 0.0), QPointF(x + 6, 0.0), QPointF(x, 9.0)])
            p.setBrush(QColor(255, 70, 70, 240))
            p.setPen(Qt.NoPen)
            p.drawPolygon(tri)
            p.setBrush(Qt.NoBrush)

    def paintEvent(self, ev):
        p = QPainter(self)
        W, H = max(1, self.width()), max(1, self.height())
        p.fillRect(0, 0, W, H, QColor(7, 9, 14))

        if self.qimg is None:
            p.setPen(QColor(72, 82, 104))
            f = QFont()
            f.setPointSizeF(10.5)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "打开音频文件以显示频谱图\n\n" "滚轮缩放 · 左键拖拽平移 · 右键拖拽创建遮罩")
            return

        if self._cache is None or self._cache.size() != QSize(W, H):
            self._render_cache()
        p.drawPixmap(0, 0, self._cache)

        # 频谱图、遮罩、八度网格、节拍线都已在 _render_cache 里画好，
        # 这里只画每帧会变的覆盖层。
        if self._hover is not None and self.n_frames > 0:
            mx, my = self._hover.x(), self._hover.y()
            midi_f = y_to_midi(my, H, self.midi_min, self.midi_max)
            midi_n = max(self.midi_min, min(self.midi_max, int(round(midi_f))))
            self._draw_harmonic_marks(p, midi_n, W, H)

            p.setPen(QPen(QColor(120, 220, 255, 160), 1, Qt.DashLine))
            p.drawLine(mx, 0, mx, H)

            y_top = midi_to_y(midi_n + 0.5, H, self.midi_min, self.midi_max)
            y_bot = midi_to_y(midi_n - 0.5, H, self.midi_min, self.midi_max)
            frame = self.view_start + mx / self.scale
            t = frame * self.hop / float(self.sr)
            f_hz = midi_to_freq(midi_n)

            beat_info = ""
            if self.bpm > 0 and self.hop > 0 and self.sr > 0:
                frames_per_beat = (60.0 / float(self.bpm)) * self.sr / float(self.hop)
                if frames_per_beat > 1e-6:
                    bi = (frame - self.beat_offset_frames) / frames_per_beat
                    b_int = int(round(bi))
                    b_num = (b_int % self.beats_per_bar) + 1
                    bar_num = (b_int // self.beats_per_bar) + 1
                    beat_info = f"  |  小节 {bar_num} 拍 {b_num}"

            name_txt = midi_name(midi_n)
            if self.chord_intervals:
                others = [midi_name(n) for n in self.chord_notes(midi_n)[1:]]
                if others:
                    name_txt = f"{midi_name(midi_n)} + {' '.join(others)}"
            txt = f" {name_txt}   {f_hz:8.2f} Hz   {t:7.3f} s{beat_info} "
            fnt = QFont()
            fnt.setPointSizeF(8.5)
            p.setFont(fnt)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(txt) + 8
            th = fm.height() + 4

            bx = mx + 10
            if bx + tw > W:
                bx = mx - tw - 10
            by = y_top - th - 6
            if by < 0:
                by = y_bot + 6
            p.fillRect(QRectF(bx, by, tw, th), QColor(16, 22, 34, 230))
            p.setPen(QPen(QColor(90, 130, 180), 1))
            p.drawRect(QRectF(bx, by, tw, th))
            p.setPen(QColor(170, 220, 255))
            p.drawText(QRectF(bx, by, tw, th), Qt.AlignCenter, txt)

        self._draw_playhead(p, W, H)

    def wheelEvent(self, e):
        if self.n_frames <= 0:
            return
        dy = e.angleDelta().y()
        if dy == 0:
            return
        factor = 1.2 ** (dy / 120.0)
        mx = e.pos().x()
        f_at = self.view_start + mx / self.scale
        self.scale = max(self._min_scale(), min(self.scale * factor, 800.0))
        self.view_start = f_at - mx / self.scale
        self._clamp_view()
        self._invalidate()

    def _midi_at_pos(self, pos):
        H = max(1, self.height())
        midi_n = int(round(y_to_midi(pos.y(), H, self.midi_min, self.midi_max)))
        return max(self.midi_min, min(self.midi_max, midi_n))

    def _frame_at_x(self, x):
        f = self.view_start + x / self.scale
        return max(0.0, min(float(self.n_frames), f))

    def _emit_hover_info(self, pos):
        if self.n_frames > 0:
            H = max(1, self.height())
            midi_n = int(round(y_to_midi(pos.y(), H, self.midi_min, self.midi_max)))
            midi_n = max(self.midi_min, min(self.midi_max, midi_n))
            notes = self.chord_notes(midi_n)
            self.hoverNote.emit(midi_n)
            # 泛音列表按和弦音展开，钢琴窗那侧也跟着亮
            self.hoverNotes.emit(self._harmonic_notes_of(notes))
            frame = self.view_start + pos.x() / self.scale
            t = frame * self.hop / float(self.sr)
            if len(notes) > 1:
                names = " ".join(midi_name(n) for n in notes)
                f_hz = midi_to_freq(notes[0])
                self.hoverInfo.emit(
                    f"{names}   {f_hz:8.2f} Hz   {t:8.3f} s   ({len(notes)} 音)")
            else:
                f_hz = midi_to_freq(midi_n)
                self.hoverInfo.emit(f"{midi_name(midi_n)}   {f_hz:8.2f} Hz   {t:8.3f} s")
        else:
            self.hoverInfo.emit("")
            self.hoverNotes.emit([])

    def mousePressEvent(self, e):
        if self.n_frames <= 0:
            return
        if e.button() == Qt.RightButton:
            if self.selected_mask is not None:
                self.selected_mask = None
                self.maskSelected.emit(-1)
            f = self._frame_at_x(e.pos().x())
            midi_n = self._midi_at_pos(e.pos())
            # 和弦模式下拖动时预览全部和弦音（松手才真正写入）
            self._mask_drag = [f, f, midi_n]
            self.setCursor(Qt.SizeHorCursor)
            self._invalidate_bg()
            return
        if e.button() == Qt.LeftButton:
            hit = self._mask_at_pos(e.pos())
            if hit is not None:
                if self.selected_mask != hit:
                    self.selected_mask = hit
                    self.maskSelected.emit(hit)
                self._press_pos = None
                self._drag_x0 = None
                self._drag_moved = False
                self.setFocus(Qt.MouseFocusReason)
                self._invalidate_bg()
                return
            if self.selected_mask is not None:
                self.selected_mask = None
                self.maskSelected.emit(-1)
                self._invalidate_bg()
            midi_n = self._midi_at_pos(e.pos())
            # 和弦模式下一次点击把所有和弦音都发出去
            self.noteTriggered.emit(midi_n)
            self._press_pos = e.pos()
            self._drag_x0 = e.pos().x()
            self._drag_view0 = self.view_start
            self._drag_moved = False
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        self._hover = e.pos()
        if self._mask_drag is not None and (e.buttons() & Qt.RightButton):
            f = self._frame_at_x(e.pos().x())
            f_start = self._mask_drag[0]
            new_end = f_start + max(0.0, f - f_start)
            if new_end > self._mask_drag[1]:
                self._mask_drag[1] = new_end
                self._invalidate_bg()
            self._emit_hover_info(e.pos())
            return
        if self._drag_x0 is not None and self._press_pos is not None:
            dx = e.pos().x() - self._drag_x0
            dy = e.pos().y() - self._press_pos.y()
            if not self._drag_moved and (abs(dx) > 4 or abs(dy) > 4):
                self._drag_moved = True
                self._cancel_follow()
            if self._drag_moved:
                self.view_start = self._drag_view0 - dx / self.scale
                self._clamp_view()
                self._invalidate()
        self._repaint()
        self._emit_hover_info(e.pos())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton and self._mask_drag is not None:
            f_start, f_end, midi_n = self._mask_drag
            self._mask_drag = None
            self.setCursor(Qt.CrossCursor)
            # 只有真正拖出了一段才算创建；右键单击不生成遮罩。
            # 阈值固定在**像素**上，不随缩放变：这样任何缩放级别下
            # 手感都一致，也符合"拖动才创建"的直觉。
            f_per_px = 1.0 / max(self.scale, 1e-9)
            if f_end - f_start < MASK_MIN_DRAG_PX * f_per_px:
                self._invalidate_bg()          # 擦掉拖动中的预览
                return
            # 和弦模式：每个和弦音各建一个独立遮罩，这样能一个一个删。
            # 选中第一个（低音）那个，和"鼠标位置就是低音"的直觉一致。
            notes = self.chord_notes(midi_n)
            first_new = len(self.masks)
            for n in notes:
                self.masks.append((f_start, f_end, n))
            self.selected_mask = first_new if notes else None
            self.maskSelected.emit(self.selected_mask if self.selected_mask is not None else -1)
            self.maskSelected.emit(self.selected_mask)
            self._invalidate_bg()      # 新遮罩要画进缓存
            return
        if e.button() == Qt.LeftButton:
            if not self._drag_moved and self._press_pos is not None and self.n_frames > 0:
                W = max(1, self.width())
                click_ratio = self._press_pos.x() / float(W)
                self._follow_ratio = max(0.05, min(0.95, click_ratio))
                self._has_manual_seek = True
                self._emit_click(e.pos())
            self._drag_x0 = None
            self._press_pos = None
            self._drag_moved = False
            self.setCursor(Qt.CrossCursor)

    def _emit_click(self, pos):
        frame = self._frame_at_x(pos.x())
        frame = max(0.0, min(float(self.n_frames - 1), frame))
        midi_n = self._midi_at_pos(pos)
        self.clicked.emit(frame, midi_n)

    def leaveEvent(self, e):
        self._hover = None
        self.hoverNote.emit(-1)
        self.hoverNotes.emit([])
        self.hoverInfo.emit("")
        self._repaint()

    def keyPressEvent(self, e):
        key = e.key()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_mask is not None and 0 <= self.selected_mask < len(self.masks):
                del self.masks[self.selected_mask]
                self.selected_mask = None
                self.maskSelected.emit(-1)
                self._invalidate_bg()
                e.accept()
                return
            e.accept()
            return
        if key == Qt.Key_Escape:
            if self.selected_mask is not None:
                self.selected_mask = None
                self.maskSelected.emit(-1)
                self._invalidate_bg()
                e.accept()
                return
        if key == Qt.Key_Left:
            self.view_start -= self.width() / self.scale * 0.15
            self._clamp_view()
            self._invalidate()
        elif key == Qt.Key_Right:
            self.view_start += self.width() / self.scale * 0.15
            self._clamp_view()
            self._invalidate()
        elif key == Qt.Key_0:
            self.fit_view()
        else:
            super().keyPressEvent(e)


# =========================================================================
# 主窗口
# =========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"频谱图分析工具  ·  {GPU_NAME}")
        self.setMinimumSize(960, 640)
        self.resize(1400, 860)
        self.setAcceptDrops(True)

        self.samples = None
        self.sr = 44100
        self.hop = 512
        self.hide_low = 0.0
        self.current_path = None
        self.db = None          # 最近一次分析结果（dB），供导出用
        self.mag = None         # 同一结果的线性幅度（未归一化）

        self.params = {
            "window_name": DEFAULT_WINDOW,
            "rows_per_semitone": DEFAULT_ROWS_PER_SEMITONE,
            "target_fps": DEFAULT_TARGET_FPS,
        }

        self.playback_samples = None
        self.wav_bytes = None
        self._wav_buffer = None
        self._wav_temp_path = None

        self._is_playing = False
        self._warming = False
        self._worker = None

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(120)
        self._hide_timer.timeout.connect(self._apply_hide_low)

        self._active_note = None
        self._note_off_timer = QTimer(self)
        self._note_off_timer.setSingleShot(True)
        self._note_off_timer.setInterval(500)
        self._note_off_timer.timeout.connect(self._stop_active_note)

        # 和弦状态：间隔表按"从低音往上数"给出；空 = 关闭
        self._custom_chord_intervals = ()
        self._chord_dlg = None

        self._playhead_timer = QTimer(self)
        self._playhead_timer.setInterval(16)
        self._playhead_timer.timeout.connect(self._tick_playhead)

        self.player = None
        self._init_media_player()

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()

        # 后台探测完成后刷新标题栏/状态栏，否则界面会一直显示 CPU。
        _notifier.changed.connect(self._on_backend_changed)
        # 万一探测在界面建好之前就结束了，这里补一次。
        if _GPU_PROBE_DONE is not None and _GPU_PROBE_DONE.is_set():
            self._on_backend_changed(GPU_STATUS)

        QTimer.singleShot(0, lambda: self.resize(1400, 860))

    def _on_backend_changed(self, name=""):
        """在主线程刷新与后端相关的文字。"""
        try:
            self.setWindowTitle(f"频谱图分析工具  ·  {GPU_NAME}")
            if hasattr(self, "lbl_right"):
                self.lbl_right.setText(f"{_backend_label()}  ·  {_midi_label()}  ·  " f"滚轮缩放 · 左键平移/发声 · 右键拖动创建遮罩")
            if self.samples is not None:
                # 已经载入过文件：重算一次当前状态的描述，避免显示陈旧信息
                self.lbl_left.setText(self._ready_text())
        except Exception as e:
            print(f"[ui] 刷新后端文字失败: {e}")

    def _ready_text(self):
        """已载入文件但还没分析完时用的简短描述。"""
        if self.current_path is None:
            return "就绪  ·  拖入音频文件或点击「打开音频」"
        name = os.path.basename(self.current_path)
        dur = len(self.samples) / float(self.sr) if self.samples is not None else 0.0
        return f"{name}   ·   {self.sr} Hz   ·   {dur:.2f} s   ·   " f"{_backend_label()}"

    def _init_media_player(self):
        if not HAS_MEDIA:
            return
        try:
            self.player = QMediaPlayer(self)
            self.player.positionChanged.connect(self._on_player_position)
            try:
                self.player.stateChanged.connect(self._on_player_state)
            except AttributeError:
                pass
        except Exception as e:
            print(f"[media] QMediaPlayer 创建失败: {e}")
            self.player = None

    def _release_media(self):
        if self.player is not None:
            try:
                self.player.stop()
            except Exception:
                pass
        if self._wav_buffer is not None:
            try:
                self._wav_buffer.close()
            except Exception:
                pass
            self._wav_buffer = None

    def _load_media_from_samples(self, samples, sr, source_path=None):
        if self.player is None:
            return
        old_tmp = self._wav_temp_path
        self._wav_temp_path = None
        self._release_media()

        try:
            wav_bytes = samples_to_wav_bytes(samples, sr)
        except Exception as e:
            print(f"[media] WAV 编码失败: {e}")
            wav_bytes = None

        if not wav_bytes:
            if source_path and source_path.lower().endswith(".wav"):
                try:
                    self.player.setMedia(QMediaContent(QUrl.fromLocalFile(source_path)))
                    self._warmup_player()
                except Exception as e:
                    print(f"[media] 回退源文件播放失败: {e}")
            if old_tmp:
                try:
                    os.remove(old_tmp)
                except Exception:
                    pass
            return

        self.wav_bytes = wav_bytes
        try:
            import tempfile

            fd, tmp = tempfile.mkstemp(prefix="wavetonepro_", suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(wav_bytes)
            self._wav_temp_path = tmp
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(tmp)))
            self._warmup_player()
            if old_tmp:
                try:
                    os.remove(old_tmp)
                except Exception:
                    pass
            return
        except Exception as e:
            print(f"[media] 临时文件写入失败，回退 QBuffer: {e}")

        try:
            from PyQt5.QtCore import QBuffer, QByteArray, QIODevice

            buf = QBuffer(self)
            buf.setData(QByteArray(wav_bytes))
            if not buf.open(QIODevice.ReadOnly):
                raise RuntimeError("QBuffer.open 失败")
            self.player.setMedia(QMediaContent(), buf)
            self._wav_buffer = buf
            self._warmup_player()
        except Exception as e:
            print(f"[media] QBuffer 回退也失败: {e}")

        if old_tmp:
            try:
                os.remove(old_tmp)
            except Exception:
                pass

    def _set_player_volume(self, value_0_1: float):
        if self.player is None:
            return
        v = max(0.0, min(1.0, float(value_0_1)))
        try:
            self.player.setVolume(int(round(v * 100)))
        except TypeError:
            self.player.setVolume(v)

    def _warmup_player(self):
        if self.player is None:
            return
        try:
            self._warming = True
            self._set_player_volume(0.0)
            self.player.play()
        except Exception as e:
            self._warming = False
            print(f"[media] 预热失败: {e}")
            return
        QTimer.singleShot(80, self._finish_warmup)

    def _finish_warmup(self):
        if not self._warming:
            return
        self._warming = False
        if self.player is None:
            return
        try:
            if not self._is_playing:
                self.player.pause()
                self.player.setPosition(0)
            self._set_player_volume(1.0)
        except Exception:
            pass

    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet("background:#000000;")
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.piano = PianoRoll(MIDI_MIN, MIDI_MAX)
        self.spec = SpectrogramView(MIDI_MIN, MIDI_MAX)

        lay.addWidget(self.piano, 0)
        lay.addWidget(self.spec, 1)

        self.spec.hoverInfo.connect(self._on_hover)
        self.spec.hoverNotes.connect(self.piano.set_highlights)
        self.spec.clicked.connect(self._on_spec_clicked)
        self.spec.noteTriggered.connect(self._play_midi_note)
        self.piano.notePressed.connect(self._on_key_pressed)
        self.spec.maskSelected.connect(self._on_mask_selected)
        self.spec.followModeChanged.connect(self._on_follow_changed_externally)

    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        tb.setStyleSheet(
            "QToolBar{background:#000000;border-bottom:1px solid #262c3a;"
            "padding:4px 8px;spacing:6px;}"
            "QToolBar QWidget{background:transparent;}"
            # 分割线：默认那条太暗在纯黑上看不见。这里做成一条明显的竖线，
            # 两侧各留 5px 间距，让分组一眼能分出来。
            "QToolBar::separator{background:#3a465e;width:2px;"
            "margin:4px 6px;border-radius:1px;}"
            "QToolButton{color:#cfd8ea;padding:3px 7px;border-radius:5px;"
            "font-size:12px;background:transparent;border:none;}"
            "QToolButton:hover{background:#1a2434;}"
            "QToolButton:checked{background:#2b5fb8;color:#ffffff;}"
            "QToolButton:disabled{color:#555c6b;}"
            "QLabel{color:#cfd8ea;font-size:11.5px;background:transparent;}"
            "QComboBox{background:#000000;color:#cfd8ea;"
            "border:1px solid #303848;border-radius:5px;"
            "padding:2px 6px;font-size:12px;}"
            "QComboBox:hover{border:1px solid #4a5878;}"
            "QComboBox::drop-down{border:none;width:16px;}"
            "QComboBox::down-arrow{image:none;border-left:4px solid transparent;"
            "border-right:4px solid transparent;border-top:5px solid #cfd8ea;"
            "width:0;height:0;margin-right:4px;}"
            "QComboBox QAbstractItemView{background:#000000;color:#cfd8ea;"
            "border:1px solid #303848;outline:none;"
            "selection-background-color:#2b5fb8;selection-color:#ffffff;}"
            "QComboBox QAbstractItemView::item{background:#000000;color:#cfd8ea;"
            "min-height:20px;padding:2px 6px;}"
            "QComboBox QAbstractItemView::item:selected{background:#2b5fb8;"
            "color:#ffffff;}"
            "QComboBox QAbstractItemView::item:hover{background:#1a2434;}"
            "QDoubleSpinBox,QSpinBox{background:#000000;color:#cfd8ea;"
            "border:1px solid #303848;border-radius:5px;"
            "padding:1px 4px;font-size:12px;}"
            "QDoubleSpinBox:focus,QSpinBox:focus{border:1px solid #4a5878;}"
            "QDoubleSpinBox::up-button,QDoubleSpinBox::down-button,"
            "QSpinBox::up-button,QSpinBox::down-button{width:12px;"
            "background:transparent;border:none;}"
            "QSlider::groove:horizontal{height:5px;background:#1a2230;"
            "border-radius:3px;}"
            "QSlider::handle:horizontal{background:#4ea1ff;width:12px;"
            "margin:-5px 0;border-radius:6px;}"
            "QSlider::sub-page:horizontal{background:#2b5fb8;border-radius:3px;}"
        )
        self.addToolBar(tb)

        act_open = QAction("打开 📁", self)
        act_open.setToolTip("打开音频 (Ctrl+O)")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_file)
        tb.addAction(act_open)

        act_params = QAction("参数 ⚙", self)
        act_params.setToolTip("修改分析参数并重新分析")
        act_params.triggered.connect(self.open_params_dialog)
        tb.addAction(act_params)

        self.act_export = QAction("导出 💾", self)
        self.act_export.setToolTip(
            "把分析结果导出为数组文件（npz / npy / csv / raw float32）\n"
            "导出的是未归一化的线性幅度，便于调试和二次处理")
        self.act_export.setShortcut("Ctrl+E")
        self.act_export.triggered.connect(self.export_result_dialog)
        self.act_export.setEnabled(False)
        tb.addAction(self.act_export)

        act_fit = QAction("适应 🔍", self)
        act_fit.setToolTip("适应窗口 (Ctrl+0)")
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(lambda: self.spec.fit_view())
        tb.addAction(act_fit)

        tb.addSeparator()

        # 播放/暂停合成一个键：图标随状态变，点击即切换
        self.act_play = QAction("播放 ▶", self)
        self.act_play.setToolTip("播放 / 暂停 (Space)")
        self.act_play.setShortcut("Space")
        self.act_play.setShortcutContext(Qt.ApplicationShortcut)
        self.act_play.triggered.connect(self._on_play_pause)
        tb.addAction(self.act_play)

        self.act_stop = QAction("停止 ⏹", self)
        self.act_stop.setToolTip("停止并回到开头")
        self.act_stop.triggered.connect(self._on_stop)
        tb.addAction(self.act_stop)

        if self.player is None:
            self.act_play.setEnabled(False)
            self.act_stop.setEnabled(False)

        self.act_follow = QAction("跟随 🎯", self)
        self.act_follow.setCheckable(True)
        self.act_follow.setChecked(False)
        self.act_follow.setToolTip("屏幕跟随播放头（拖动频谱自动取消）")
        self.act_follow.toggled.connect(self._on_follow_toggled)
        tb.addAction(self.act_follow)

        tb.addSeparator()

        # 谱面设置：音域 / 外观 / 小节线 都收进这个窗口，工具栏只留一个入口
        self.act_spec_cfg = QAction("谱面设置 ⚙", self)
        self.act_spec_cfg.setToolTip(
            "打开谱面设置窗口\n"
            "钢琴窗显示音域 · 主题与泛音 · 小节线/节拍线 · 亮度滤镜")
        self.act_spec_cfg.triggered.connect(self.open_spectrum_settings)
        tb.addAction(self.act_spec_cfg)

        tb.addSeparator()

        self.act_clear_mask = QAction("清空遮罩 🧹", self)
        self.act_clear_mask.setToolTip("清除所有遮罩")
        self.act_clear_mask.triggered.connect(self._on_clear_masks)
        tb.addAction(self.act_clear_mask)

        # ---- 和弦 ----
        tb.addWidget(QLabel("和弦"))
        self.cb_chord = QComboBox()
        self.cb_chord.setToolTip(
            "和弦模式：鼠标位置作为**低音**，一次标记/遮罩/发声多个音。\n"
            "间隔都是从低音往上数的，所以转位不改变低音。")
        self.cb_chord.setMinimumWidth(150)
        self.cb_chord.addItem(CHORD_NONE, None)
        for label, max_inv, iv in CHORDS:
            self.cb_chord.addItem(label, iv)
        self.cb_chord.currentIndexChanged.connect(self._on_chord_changed)
        tb.addWidget(self.cb_chord)

        tb.addWidget(QLabel("转位"))
        self.sp_chord_inv = QSpinBox()
        self.sp_chord_inv.setRange(0, 0)
        self.sp_chord_inv.setFixedWidth(40)
        self.sp_chord_inv.setToolTip(
            "转位：把最低的 n 个音依次往上挪一个八度。低音位置不变。")
        self.sp_chord_inv.valueChanged.connect(self._on_chord_changed)
        tb.addWidget(self.sp_chord_inv)

        self.act_chord_custom = QAction("自定义和弦 🎹", self)
        self.act_chord_custom.setToolTip("打开两个八度的钢琴窗，自己点出和弦形状")
        self.act_chord_custom.setCheckable(True)
        self.act_chord_custom.toggled.connect(self._on_chord_custom_toggled)
        tb.addAction(self.act_chord_custom)

        # 泛音数量：它的效果要靠鼠标悬停看，放对话框里没法预览，所以留在工具栏
        tb.addWidget(QLabel("泛音"))
        self.cb_harm = QComboBox()
        self.cb_harm.setToolTip("悬停时高亮的泛音数量（0 = 只显示基音）")
        self.cb_harm.setFixedWidth(60)
        for n in range(6):
            self.cb_harm.addItem(f"{n} 个" if n else "基音", n)
        self.cb_harm.currentIndexChanged.connect(
            lambda i: self._on_harmonics_changed(self.cb_harm.itemData(i)))
        tb.addWidget(self.cb_harm)

        tb.addSeparator()

        self.act_step = QAction("阶梯滤镜 ▤", self)
        self.act_step.setCheckable(True)
        self.act_step.setChecked(False)
        self.act_step.setToolTip(
            "后期滤镜：把每个半音压成一条平带，半音之间出现硬边界（阶梯感）\n"
            "只影响显示，不改动分析结果\n"
            "参数在右侧「调参」里")
        self.act_step.toggled.connect(self._on_step_filter_toggled)
        tb.addAction(self.act_step)

        self.act_step_cfg = QAction("阶梯参数 ⚙", self)
        self.act_step_cfg.setToolTip("打开阶梯滤镜参数窗口（实时预览）")
        self.act_step_cfg.triggered.connect(self.open_filter_config)
        tb.addAction(self.act_step_cfg)

        tb.addSeparator()

        # 亮度滤镜没有工具栏控件，它的滑块在「谱面设置」窗口里。
        # 这里只维护状态：窗口创建滑块后会把引用交给 _volume_slider，
        # 关闭后主窗口仍持有它（滑块以对话框为 parent，不会被删）。
        self._volume_slider = None
        self._volume_percent = 35
        self.hide_low = self._volume_percent / 100.0
        self.spec.set_hide_low(self.hide_low)

    def _build_statusbar(self):
        sb = QStatusBar()
        sb.setStyleSheet("QStatusBar{background:#000000;color:#cfd8ea;" "border-top:1px solid #262c3a;}" "QStatusBar::item{border:none;}" "QStatusBar QLabel{color:#cfd8ea;background:transparent;}")
        self.setStatusBar(sb)

        self.lbl_left = QLabel("就绪  ·  拖入音频文件或点击「打开音频」")
        self.lbl_left.setStyleSheet("color:#cfd8ea;background:transparent;")
        self.lbl_center = QLabel("")
        midi_txt = _midi_label()
        self.lbl_right = QLabel(f"{_backend_label()}  ·  {midi_txt}  ·  " f"滚轮缩放 · 左键平移/发声 · 右键拖动创建遮罩")
        self.lbl_right.setStyleSheet("color:#8b96ad;background:transparent;")
        self.lbl_center.setStyleSheet("color:#9fc5ff;font-family:Consolas,Menlo,monospace;" "background:transparent;")

        sb.addWidget(self.lbl_left, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(180)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar{background:#000000;color:#cfd8ea;" "border:1px solid #303848;border-radius:5px;" "text-align:center;font-size:10.5px;}" "QProgressBar::chunk{background:#2b5fb8;border-radius:4px;}"
        )
        sb.addPermanentWidget(self.progress_bar, 0)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setFixedHeight(20)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet("QPushButton{background:#000000;color:#ffb4b4;" "border:1px solid #5a2b34;border-radius:5px;" "padding:0 8px;font-size:11px;}" "QPushButton:hover{background:#3a2028;}")
        # clicked 会带一个 checked: bool 参数，直接接 _cancel_analysis 会把它
        # 当成 wait，导致 `if wait:` 分支从按钮永远走不到。这里显式丢弃。
        self.btn_cancel.clicked.connect(lambda _checked=False: self._cancel_analysis(wait=False))
        sb.addPermanentWidget(self.btn_cancel, 0)

        sb.addPermanentWidget(self.lbl_center, 0)
        sb.addPermanentWidget(self.lbl_right, 0)

    def _on_hover(self, text):
        self.lbl_center.setText(text)

    def _on_key_pressed(self, midi):
        self._play_midi_note(midi)
        f = midi_to_freq(midi)
        self.lbl_left.setText(f"琴键  {midi_name(midi)}    MIDI {midi}    {f:.2f} Hz")

    def _on_mask_selected(self, idx):
        if idx < 0:
            cur = self.lbl_left.text()
            if "已选中遮罩" in cur:
                self.lbl_left.setText("已取消遮罩选择")
            return
        self.lbl_left.setText(f"已选中遮罩 #{idx + 1}  ·  按 Delete/Backspace 删除，Esc 取消")

    def _on_follow_toggled(self, on):
        self.spec.set_follow_mode(bool(on))
        if on:
            self.lbl_left.setText("跟随模式已开启")
        else:
            self.lbl_left.setText("跟随模式已关闭")

    def _on_follow_changed_externally(self, on):
        if self.act_follow.isChecked() != on:
            self.act_follow.setChecked(on)
        self.lbl_left.setText("跟随模式已开启" if on else "跟随模式已关闭（拖动已取消）")

    # ------------------------------------------------------------------
    # 谱面设置
    # ------------------------------------------------------------------
    def _current_spectrum_settings(self):
        """当前设置快照。亮度滤镜读缓存值，不碰窗口里那个可能已销毁的滑块。

        泛音数量不在其中：它由工具栏的下拉框负责，放在窗口里没法用悬停预览。
        """
        return {
            "midi_min": int(self.spec.midi_min),
            "midi_max": int(self.spec.midi_max),
            "colormap": self.spec.cmap,
            "show_beats": bool(self.spec.show_beats),
            "bpm": float(self.spec.bpm),
            "beats_per_bar": int(self.spec.beats_per_bar),
            "hide_low": float(self.hide_low),
            "volume_percent": int(self._volume_percent),
        }

    def _apply_spectrum_settings(self, s):
        """把一份设置推到各个视图上。窗口里改一处就会走一次。

        亮度滤镜在这里只负责同步滑块位置（静默），实际生效由
        sld_hide.valueReleased -> _hide_timer -> _apply_hide_low 负责。
        """
        lo = int(s.get("midi_min", MIDI_MIN))
        hi = int(s.get("midi_max", MIDI_MAX))
        if (lo, hi) != (self.spec.midi_min, self.spec.midi_max):
            self.piano.set_midi_range(lo, hi)
            self.spec.set_midi_range(lo, hi)
        self.spec.set_colormap(s.get("colormap", self.spec.cmap))
        self.spec.set_bpm(float(s.get("bpm", self.spec.bpm)))
        self.spec.set_beats_per_bar(int(s.get("beats_per_bar", self.spec.beats_per_bar)))
        self.spec.set_show_beats(bool(s.get("show_beats", self.spec.show_beats)))
        pct = s.get("volume_percent")
        if pct is not None:
            self._volume_percent = int(pct)
            sl = self._volume_slider
            if sl is not None:
                try:
                    if sl.value() != self._volume_percent:
                        sl.setValueSilently(self._volume_percent)
                except RuntimeError:
                    self._volume_slider = None      # 底层对象已销毁
        self.lbl_left.setText(
            f"音域 {midi_name(lo)}–{midi_name(hi)}（{hi - lo + 1} 个半音）"
            f"  ·  {s.get('colormap', self.spec.cmap)}"
            f"  ·  {'节拍线开' if s.get('show_beats', True) else '节拍线关'}")

    def open_spectrum_settings(self):
        """打开谱面设置窗口（音域 / 外观 / 小节线）。"""
        original = self._current_spectrum_settings()
        has_data = self.spec.db is not None

        def preview(s):
            self._apply_spectrum_settings(s)

        dlg = SpectrumSettingsDialog(original, on_change=preview,
                                     original=original, has_data=has_data,
                                     parent=self)
        sl = dlg.sld_hide
        # 直接把信号连到主窗口：窗口销毁时 Qt 会自己断开，不会留下野连接
        sl.valueReleased.connect(self._on_hide_slider_changed)
        try:
            ok = dlg.exec_() == QDialog.Accepted
        finally:
            # 先把滑块要过来、再销毁对话框。顺序不能反：
            # deleteLater 之后还挂在对话框名下的子控件会被一起回收，
            # 归到主窗口名下它才能安全活到程序结束。
            sl.hide()
            sl.setParent(self)
            old = self._volume_slider
            if old is not None and old is not sl:
                old.deleteLater()       # 上一次留下的滑块，回收掉
            self._volume_slider = sl
            dlg.deleteLater()
        if ok:
            self._apply_spectrum_settings(dlg.get_settings())
        self.spec.setFocus()

    def _on_spec_clicked(self, frame_pos, midi_note):
        self.spec.set_playhead_frame(frame_pos)
        if self.player is not None and self.hop > 0 and self.sr > 0:
            ms = int(frame_pos * self.hop * 1000.0 / self.sr)
            was_playing = self._is_playing
            try:
                if was_playing:
                    self.player.pause()
                self.player.setPosition(ms)
                if was_playing:
                    self.player.play()
            except Exception as e:
                print(f"[media] setPosition 失败: {e}")
            self.lbl_left.setText(f"跳转至  {frame_pos * self.hop / self.sr:.3f} s   ·   " f"{midi_name(midi_note)}")

    def _play_midi_note(self, midi):
        """和弦模式下一次点击把和弦的所有音一起发出去。"""
        if not MIDI_AVAILABLE:
            return
        self._stop_active_note()
        notes = self.spec.chord_notes(midi)
        for n in notes:
            midi_note_on(n, vel=90)
        self._active_note = list(notes)
        self._note_off_timer.start()

    def _stop_active_note(self):
        if self._active_note is None:
            return
        # 兼容早期只存单个 int 的情况
        notes = (self._active_note if isinstance(self._active_note, (list, tuple))
                 else [self._active_note])
        for n in notes:
            midi_note_off(n)
        self._active_note = None

    def _on_hide_slider_changed(self, v):
        """松手后才走到这里（ReleaseSlider 只发 valueReleased）。"""
        self._volume_percent = int(v)
        self._hide_timer.start()

    def _apply_hide_low(self):
        v = self._volume_percent
        new_val = v / 100.0
        if abs(new_val - self.hide_low) < 1e-6:
            return
        self.hide_low = new_val
        self.spec.set_hide_low(self.hide_low)

    def _on_clear_masks(self):
        self.spec.clear_masks()
        self.lbl_left.setText("已清除所有遮罩")

    def _on_harmonics_changed(self, n):
        self.spec.set_harmonics(int(n))
        self.lbl_left.setText(f"泛音高亮：{'只显示基音' if not n else str(n) + ' 个泛音'}")

    # ------------------------------------------------------------------
    # 和弦
    # ------------------------------------------------------------------
    def _current_chord(self):
        """返回 (显示名, 间隔表)。间隔表为空表示关闭。

        "自定义"这一项在下拉框里存的是 None，它的间隔来自
        _custom_chord_intervals，所以要先按显示文本判断。
        """
        text = self.cb_chord.currentText()
        if text == CUSTOM_CHORD_NAME:
            iv = self._custom_chord_intervals or ()
            return (CUSTOM_CHORD_NAME if iv else CHORD_NONE), iv
        iv = self.cb_chord.currentData()
        return text, (tuple(iv) if iv else ())

    def _on_chord_changed(self, *_a):
        """下拉框 / 转位框变化。

        程序改写下拉框时都套了 blockSignals，所以走到这里一定是用户操作：
        如果选了预设项，就顺手退出自定义模式。
        """
        if (self.act_chord_custom.isChecked()
                and self.sender() is self.cb_chord
                and self.cb_chord.currentText() != CUSTOM_CHORD_NAME):
            self.act_chord_custom.setChecked(False)
            return                       # setChecked 会触发 _apply_chord
        self._apply_chord()

    def _apply_chord(self):
        name, iv = self._current_chord()
        max_inv = max(0, len(iv) - 1)
        if self.sp_chord_inv.maximum() != max_inv:
            self.sp_chord_inv.blockSignals(True)
            self.sp_chord_inv.setRange(0, max_inv)
            if self.sp_chord_inv.value() > max_inv:
                self.sp_chord_inv.setValue(0)
            self.sp_chord_inv.blockSignals(False)
        inv = int(self.sp_chord_inv.value()) if iv else 0
        if not iv:
            inv = 0
        self.spec.set_chord(iv, inv)
        if iv:
            shape = "-".join(str(x) for x in chord_intervals(iv, inv))
            shown = name + (f" 第{inv}转位" if inv else "")
            self.lbl_left.setText(
                f"和弦模式：{shown}  ·  低音为鼠标位置  ·  半音间隔 {shape}")
        else:
            self.lbl_left.setText("和弦模式已关闭（单音）")

    def _on_chord_custom_toggled(self, on):
        """打开/关闭自定义和弦窗口。

        关掉窗口后"自定义"那一项会留在下拉框里并被选中，所以点好的和弦继续
        生效 —— 下拉框始终代表"当前正在用的和弦"。想回预设直接选预设即可。
        """
        if on:
            if self._chord_dlg is None:
                self._chord_dlg = ChordBuilderDialog(
                    [], on_change=self._on_custom_chord_changed, parent=self)
            self._custom_chord_intervals = self._chord_dlg.intervals()
            self._chord_dlg.show()
            self._chord_dlg.raise_()
            self._chord_dlg.activateWindow()
            self._sync_chord_combo_label()
        else:
            if self._chord_dlg is not None:
                self._custom_chord_intervals = self._chord_dlg.intervals()
                self._chord_dlg.hide()
                # 选中"自定义"这一项，让和弦保持生效
                i = self.cb_chord.findText(CUSTOM_CHORD_NAME)
                if i >= 0:
                    self.cb_chord.blockSignals(True)
                    self.cb_chord.setCurrentIndex(i)
                    self.cb_chord.blockSignals(False)
        self._apply_chord()

    def _on_custom_chord_changed(self, notes):
        """自定义窗口里点了音：立刻生效，方便当场看效果。"""
        if not self.act_chord_custom.isChecked():
            return
        self._custom_chord_intervals = self._intervals_of(notes)
        self._sync_chord_combo_label()
        self._apply_chord()

    def _intervals_of(self, notes):
        notes = sorted(int(n) for n in notes)
        if not notes:
            return ()
        base = notes[0]
        return tuple(n - base for n in notes)
    def _sync_chord_combo_label(self):
        """让下拉框反映当前的"自定义"和弦状态。

        有自定义和弦就把"自定义"插进去并选中；没有就退回第一项（无）。
        """
        if not self.act_chord_custom.isChecked():
            return
        has = bool(self._custom_chord_intervals)
        i = self.cb_chord.findText(CUSTOM_CHORD_NAME)
        self.cb_chord.blockSignals(True)
        try:
            if has and i < 0:
                self.cb_chord.insertItem(1, CUSTOM_CHORD_NAME, None)
                i = 1
            elif not has and i >= 0:
                self.cb_chord.removeItem(i)
                i = -1
            if has:
                self.cb_chord.setCurrentIndex(i)
            else:
                self.cb_chord.setCurrentIndex(0)
        finally:
            self.cb_chord.blockSignals(False)

    def _on_step_filter_toggled(self, on):
        self.spec.set_step_filter(bool(on))
        if on:
            self._describe_filter()
        else:
            self.lbl_left.setText("阶梯滤镜已关闭")

    def _describe_filter(self):
        c = self.spec.step_config
        red = {"mean": "平均", "midmax": f"去{c['trim']}行取最大"}
        cur = {"none": "无曲线", "knee": f"knee{c['knee_th']:.2f}×{c['knee_gain']:.2f}",
               "power": f"幂{c['pow_gamma']:.2f}@{c['pow_pivot']:.2f}",
               "sigmoid": f"S拐点{c['sig_center']:.2f}/宽{c['sig_k']:.2f}"}
        self.lbl_left.setText(
            f"阶梯滤镜已开启 · {red.get(c['reduce'], c['reduce'])} + "
            f"{cur.get(c['curve'], c['curve'])}（仅显示）")

    def open_filter_config(self):
        """打开滤镜参数窗口，带实时预览。"""
        if self.db is None:
            QMessageBox.information(self, "暂无数据", "请先打开并分析一个音频文件。")
            return
        if not self.spec.step_filter:
            self.act_step.setChecked(True)      # 打开滤镜才能看到效果
        original = dict(self.spec.step_config)

        def preview(cfg):
            self.spec.set_step_config(cfg)
            self._describe_filter()

        dlg = FilterConfigDialog(self.spec.step_config, on_change=preview,
                                 original=original, parent=self,
                                 rows_per_semitone=self.spec.rows_per_semitone,
                                 on_trim_user=self.spec._set_trim_user)
        try:
            ok = dlg.exec_() == QDialog.Accepted
        finally:
            dlg.deleteLater()
        if ok:
            self.spec.set_step_config(dlg.get_config())
        self._describe_filter()
        self.spec.setFocus()

    def _set_play_icon(self, playing):
        """播放键的图标跟着状态走（播放中显示暂停）。"""
        try:
            self.act_play.setText("暂停 ⏸" if playing else "播放 ▶")
        except Exception:
            pass

    def _on_play_pause(self):
        """一个键切换播放/暂停。"""
        if self._is_playing:
            self._on_pause()
        else:
            self._on_play()

    def _on_play(self):
        if self.player is None or self.current_path is None:
            return
        try:
            self._set_player_volume(1.0)
            self.player.play()
        except Exception as e:
            print(f"[media] play 失败: {e}")
            return
        self._warming = False
        self._is_playing = True
        self._set_play_icon(True)
        try:
            self._apply_position_ms(self.player.position())
        except Exception:
            pass
        self._playhead_timer.start()

    def _on_pause(self):
        if self.player is None:
            return
        try:
            self.player.pause()
        except Exception as e:
            print(f"[media] pause 失败: {e}")
        self._is_playing = False
        self._set_play_icon(False)
        self._playhead_timer.stop()

    def _on_stop(self):
        if self.player is None:
            return
        try:
            self.player.pause()
            self.player.setPosition(0)
        except Exception as e:
            print(f"[media] stop(soft) 失败: {e}")
        self._is_playing = False
        self._set_play_icon(False)
        self._playhead_timer.stop()
        self.spec.reset_follow_state()
        self.spec.set_playhead_frame(0.0)
        self.lbl_left.setText("已停止，播放头归零")

    def _tick_playhead(self):
        if self.player is None or not self._is_playing:
            return
        try:
            ms = self.player.position()
        except Exception:
            return
        self._apply_position_ms(ms)

    def _on_player_position(self, ms):
        if self._warming:
            return
        self._apply_position_ms(ms)

    def _apply_position_ms(self, ms):
        if self.hop <= 0 or self.sr <= 0:
            return
        frame = (ms / 1000.0) * self.sr / float(self.hop)
        self.spec.set_playhead_frame(frame)

    def _on_player_state(self, state):
        if self._warming:
            return
        try:
            if QMediaPlayer is not None and hasattr(QMediaPlayer, "PlayingState"):
                self._is_playing = state == QMediaPlayer.PlayingState
                self._set_play_icon(self._is_playing)
        except Exception:
            pass

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开音频文件", "", "音频文件 (*.wav *.flac *.ogg *.mp3 *.m4a *.aiff *.aif);;所有文件 (*)")
        if path:
            self.load_path(path)

    def open_params_dialog(self):
        dlg = AnalysisParamsDialog(self.params, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.params = dlg.get_params()
        if self.samples is not None:
            self._rebuild(fit=True)
            self.lbl_left.setText("参数已更新，正在重新分析…")

    def load_path(self, path):
        self._cancel_analysis(wait=True)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            samples, sr = load_audio(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "无法加载", str(e))
            return
        QApplication.restoreOverrideCursor()

        dlg = AnalysisParamsDialog(self.params, self)
        if dlg.exec_() != QDialog.Accepted:
            self.lbl_left.setText("已取消加载")
            return
        self.params = dlg.get_params()

        self.samples = to_mono(samples)
        self.sr = sr
        self.current_path = path
        self._is_playing = False

        self.playback_samples = samples
        self._load_media_from_samples(samples, sr, source_path=path)

        name = os.path.basename(path)
        dur = len(self.samples) / float(sr)
        ch_txt = ""
        try:
            s = np.asarray(samples)
            if s.ndim == 2:
                ch_txt = f"   ·   {s.shape[1]}ch → 播放双声道"
            else:
                ch_txt = "   ·   1ch → 播放双声道"
        except Exception:
            pass
        self.lbl_left.setText(f"{name}   ·   {sr} Hz   ·   {dur:.2f} s{ch_txt}" f"   ·   {self.params['window_name']}" f"   ·   {self.params['rows_per_semitone']} 行/半音   ·   正在分析…")

        self._rebuild(fit=True)

    def _cancel_analysis(self, wait=False):
        w = self._worker
        if w is not None and w.isRunning():
            w.cancel()
            if wait:
                if not w.wait(3000):
                    print("[worker] 分析线程未在 3 秒内退出，仍继续")
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool):
        if busy:
            self.progress_bar.setVisible(True)
            self.btn_cancel.setVisible(True)
            self.act_play.setEnabled(False)
            self.act_stop.setEnabled(False)
        else:
            self.progress_bar.setVisible(False)
            self.btn_cancel.setVisible(False)
            if self.player is not None:
                self.act_play.setEnabled(True)
                self.act_stop.setEnabled(True)

    def _on_analysis_progress(self, pct, text):
        self.progress_bar.setValue(max(0, min(100, int(pct))))
        self.lbl_center.setText(text)

    def _on_analysis_failed(self, msg):
        self._set_busy(False)
        if msg == "__cancelled__":
            self.lbl_left.setText("分析已取消")
            self.lbl_center.setText("")
        else:
            QMessageBox.critical(self, "分析失败", msg)
            self.lbl_left.setText("分析失败")
            self.lbl_center.setText("")

    def _on_analysis_done(self, db, hop, mag=None):
        self._set_busy(False)
        self.hop = hop
        self.db = db
        self.mag = mag
        self.spec.hop = hop
        self.spec.sr = self.sr
        self.spec.set_data(db, hop, self.sr, fit=True,
                           rows_per_semitone=self.params["rows_per_semitone"])
        self.spec.set_playhead_frame(0.0)
        try:
            self.act_export.setEnabled(True)
        except Exception:
            pass

        n_frames, n_rows = db.shape
        backend = _backend_short()
        self.lbl_left.setText(f"就绪  ·  {backend}  ·  " f"{n_frames} 帧 × {n_rows} 行  ·  hop {hop} " f"({self.sr / hop:.1f} fps)  ·  " f"{self.params['window_name']}  ·  {self.params['rows_per_semitone']} 行/半音")
        self.lbl_center.setText("")
        self.spec.setFocus()

    def _rebuild(self, fit=True):
        if self.samples is None:
            return
        self._cancel_analysis(wait=True)
        # 确保后端已定下来，这样状态栏文字和实际计算用的是同一个后端
        _ensure_backend_ready()

        backend = _backend_short()
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self.lbl_left.setText(f"多分辨率谱重分配 · {self.params['window_name']} · " f"{backend} 计算中…")

        worker = AnalysisWorker(self.samples, self.sr, self.params, self)
        worker.progress.connect(self._on_analysis_progress)
        worker.done.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))

        self._worker = worker
        worker.start()

    def _on_worker_finished(self, w):
        if self._worker is w:
            self._worker = None

    # ------------------------------------------------------------------
    # 导出分析结果（数组）
    # ------------------------------------------------------------------
    def export_result_dialog(self):
        """把最近一次分析结果导出成数组文件。"""
        if self.db is None:
            QMessageBox.information(self, "无可导出数据", "请先打开并分析一个音频文件。")
            return

        base = "analysis"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
        start = os.path.join(os.path.dirname(self.current_path or ""), base + ".npz")

        flt = ("NumPy 压缩包 (*.npz);;NumPy 单数组 (*.npy);;"
               "CSV 文本 (*.csv);;原始 float32 (*.bin);;所有文件 (*)")
        path, chosen = QFileDialog.getSaveFileName(self, "导出分析结果（数组）", start, flt)
        if not path:
            return

        fmt = {"NumPy 压缩包 (*.npz)": "npz",
               "NumPy 单数组 (*.npy)": "npy",
               "CSV 文本 (*.csv)": "csv",
               "原始 float32 (*.bin)": "raw(f32)"}.get(chosen)
        if fmt is None:
            ext = os.path.splitext(path)[1].lower()
            fmt = {"npz": "npz", "npy": "npy", "csv": "csv",
                   "bin": "raw(f32)", "raw": "raw(f32)", "f32": "raw(f32)"}.get(ext, "npz")
        # 保证扩展名和格式一致，避免写出 .npz 后缀的 csv
        want_ext = {"npz": ".npz", "npy": ".npy", "csv": ".csv", "raw(f32)": ".bin"}[fmt]
        if not path.lower().endswith(want_ext):
            path += want_ext

        # mag 是未归一化的线性幅度（原始数据）；u8 只是显示用的量化结果
        mag = self.mag
        if mag is None:
            # 兜底：从 dB 反推（会受 floor 裁剪影响，仅用于老数据）
            mag = np.power(np.float64(10.0),
                           np.asarray(self.db, dtype=np.float64) / 20.0).astype(np.float32)
        payload = export_payload(
            mag, self.db, self.hop, self.sr, params=self.params,
            source_path=self.current_path, hide_low=self.hide_low,
            colormap=getattr(self.spec, "cmap", None),
            u8=getattr(self.spec, "u8", None),
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            out = export_result(path, payload, fmt)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "导出失败", f"{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()

        try:
            size = os.path.getsize(out)
            size_txt = (f"{size/1048576:.1f} MB" if size >= 1048576
                        else f"{size/1024:.0f} KB")
        except OSError:
            size_txt = "?"
        n_frames, n_rows = self.db.shape
        self.lbl_left.setText(
            f"已导出 {os.path.basename(out)}   ·   {fmt}   ·   "
            f"{n_frames} 帧 × {n_rows} 行   ·   {size_txt}")
        self.lbl_center.setText("")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path):
                self.load_path(path)
                break

    def closeEvent(self, e):
        self._cancel_analysis(wait=True)
        if self.player is not None:
            for sig_name in (
                "positionChanged",
                "stateChanged",
                "mediaStatusChanged",
                "error",
                "durationChanged",
            ):
                try:
                    getattr(self.player, sig_name).disconnect()
                except Exception:
                    pass
            try:
                self.player.stop()
            except Exception:
                pass
        if self._wav_temp_path:
            try:
                os.remove(self._wav_temp_path)
            except Exception:
                pass
            self._wav_temp_path = None
        self._wav_buffer = None
        super().closeEvent(e)


# =========================================================================
# 入口
# =========================================================================
def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 后端探测放到后台线程：界面立刻出现，探测结果在首次分析前生效。
    # 探测阻塞/超时都不会影响程序启动。
    _start_gpu_probe()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    app.setStyleSheet(
        "QWidget{background:#000000;color:#cfd8ea;}"
        "QMainWindow{background:#000000;}"
        "QToolTip{background:#000000;color:#cfd8ea;"
        "border:1px solid #303848;padding:3px 6px;}"
        "QMenu{background:#000000;color:#cfd8ea;border:1px solid #303848;}"
        "QMenu::item:selected{background:#2b5fb8;color:#ffffff;}"
        "QDialog{background:#0a0d13;color:#cfd8ea;}"
        "QDialog QLabel{color:#cfd8ea;}"
        "QDialog QComboBox{background:#000000;color:#cfd8ea;"
        "border:1px solid #303848;border-radius:5px;padding:2px 6px;}"
        "QDialog QPushButton{background:#131a26;color:#cfd8ea;"
        "border:1px solid #303848;border-radius:5px;padding:4px 12px;}"
        "QDialog QPushButton:hover{background:#1a2434;}"
        "QScrollBar:vertical{background:#000000;width:10px;margin:0;}"
        "QScrollBar::handle:vertical{background:#2a3346;border-radius:5px;min-height:20px;}"
        "QScrollBar::handle:vertical:hover{background:#3a4560;}"
        "QScrollBar:horizontal{background:#000000;height:10px;margin:0;}"
        "QScrollBar::handle:horizontal{background:#2a3346;border-radius:5px;min-width:20px;}"
        "QScrollBar::handle:horizontal:hover{background:#3a4560;}"
        "QScrollBar::add-line,QScrollBar::sub-line{background:transparent;height:0;width:0;}"
        "QScrollBar::add-page,QScrollBar::sub-page{background:transparent;}"
    )

    _init_midi()
    if MIDI_AVAILABLE:
        print(f"[midi] 已连接输出: {MIDI_PORT_NAME}  (backend={MIDI_BACKEND})")
    else:
        print("[midi] 未检测到可用输出（点击频谱不会发声）")

    win = MainWindow()
    win.show()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win.load_path(sys.argv[1])

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
