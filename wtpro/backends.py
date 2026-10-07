"""计算后端（CPU / CuPy）与 MIDI 输出的探测、切换与状态查询。

注意 HAS_GPU / xp 是**可变**的模块级状态：使用方必须持有本模块
（`from wtpro import backends`）后按 backends.xp 取值，
不能用 `from ... import xp` 取快照，否则后端切换后拿到旧值。"""

import os
import sys
import threading
import json
import tempfile

import numpy as np

from wtpro.qt import _notifier

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


MIDI_AVAILABLE = False


MIDI_BACKEND = None


MIDI_PORT_NAME = ""


_midi_out = None




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


