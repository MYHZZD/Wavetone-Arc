import sys
import os
import io
import math
import wave

import numpy as np

# =========================================================================
# 后端探测
# =========================================================================
HAS_GPU = False
GPU_NAME = "CPU (NumPy)"
xp = np

try:
    import cupy as _cp

    _t = _cp.zeros(1)
    _t += 1
    del _t
    xp = _cp
    HAS_GPU = True
    try:
        _n = _cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(_n, bytes):
            _n = _n.decode("utf-8", "ignore")
        GPU_NAME = f"GPU ({_n})"
    except Exception:
        GPU_NAME = "GPU (CUDA)"
except Exception:
    HAS_GPU = False
    GPU_NAME = "CPU (NumPy)"
    xp = np


# =========================================================================
# MIDI 输出初始化
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
    Qt, QRectF, QPointF, QSize, pyqtSignal, QTimer, QUrl, QThread,
)
from PyQt5.QtGui import QImage, QPainter, QColor, QPen, QFont, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QFileDialog, QLabel,
    QComboBox, QSizePolicy, QMessageBox, QToolBar, QAction, QSlider,
    QStatusBar, QProgressBar, QPushButton, QDoubleSpinBox, QSpinBox,
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
# 常量
# =========================================================================
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BLACK_PC = {1, 3, 6, 8, 10}

MIDI_MIN = 21
MIDI_MAX = 108
ROWS_PER_SEMITONE = 12

DB_FLOOR = -100.0
DISPLAY_GAMMA = 1.0

FFT_MIN = 1024
FFT_MAX = 262144
TARGET_FPS = 150

SPLAT_K = 2
SPLAT_SIGMA_LOW = 0.6
SPLAT_SIGMA_HIGH = 1.0

ENERGY_THRESHOLD_DB = -60.0

TIME_SPLAT_K = 2
TIME_SPLAT_SIGMA = 0.6
TIME_INV_SIGMA_SQ_HALF = -0.5 / (TIME_SPLAT_SIGMA * TIME_SPLAT_SIGMA)

GROUP_OVERLAP = 4

F32 = np.float32
F64 = np.float64


# =========================================================================
# 颜色映射
# =========================================================================
MAGMA_STOPS = [
    (0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129),
    (181, 54, 122), (229, 80, 100), (251, 135, 97), (254, 194, 135), (252, 253, 191),
]
INFERNO_STOPS = [
    (0, 0, 4), (22, 11, 57), (66, 10, 104), (106, 23, 110), (147, 38, 103),
    (188, 55, 84), (221, 81, 58), (243, 120, 25), (252, 165, 10), (246, 215, 70), (252, 255, 164),
]
VIRIDIS_STOPS = [
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142), (38, 130, 142),
    (31, 158, 137), (53, 183, 121), (109, 205, 89), (180, 222, 44), (253, 231, 37),
]
ICE_STOPS = [
    (2, 4, 10), (8, 22, 50), (10, 52, 96), (8, 92, 140), (20, 140, 175),
    (70, 185, 205), (140, 220, 230), (210, 245, 250), (255, 255, 255),
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
# FFT 分组
# =========================================================================
def _next_pow2(n):
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _prev_pow2(n):
    n = max(1, int(n))
    return 1 << (n.bit_length() - 1)


def _plan_groups(sr, midi_min, midi_max, min_fft=FFT_MIN, max_fft=FFT_MAX, group_semitones=6, overlap=GROUP_OVERLAP):
    ratio = 2.0 ** (1.0 / 12.0) - 1.0
    groups = []
    m = midi_min
    while m <= midi_max:
        m_end = min(m + group_semitones, midi_max + 1)
        delta_f = midi_to_freq(m) * ratio
        N = _next_pow2(6.0 * sr / delta_f)
        N = int(min(max(N, min_fft), max_fft))
        hop_nat = max(1, _prev_pow2(max(1, N // overlap)))
        groups.append((m, m_end, N, hop_nat))
        m = m_end
    return groups


# =========================================================================
# 取消异常
# =========================================================================
class AnalysisCancelled(Exception):
    pass


# =========================================================================
# STFT
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
# 阶段 1：谱重分配
# =========================================================================
def _reassign_chunk(
    X, N, hop, sr, f_k, n_rows, midi_min, midi_max,
    rows_per_semitone, group_grid, i0_group, k_lo=0, k_hi=None,
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

    dphase = (phase[2:] - phase[:-2]) * 0.5
    del phase

    mag_mid = mag[1:-1]
    del mag

    k_idx = xp.arange(k_lo, k_hi, dtype=xp.float64)
    dphase -= (2.0 * xp.pi * float(hop) / float(N)) * k_idx[None, :]
    dphase += xp.pi
    xp.mod(dphase, 2.0 * xp.pi, out=dphase)
    dphase -= xp.pi
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
# 阶段 2：分组粗网格 → 公共网格（时间高斯 splat ）
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
# 分块规划
# =========================================================================
def _plan_chunks(sr, n_samples, groups):
    plans = []
    for m_start, m_end, N, hop_g in groups:
        n_frames_g = n_samples // hop_g + 1
        n_bins = N // 2 + 1
        f_k_np = np.arange(n_bins, dtype=np.float64) * (float(sr) / float(N))

        k_lo = int(np.searchsorted(f_k_np, midi_to_freq(m_start), side="left"))
        k_hi = int(np.searchsorted(f_k_np, midi_to_freq(m_end), side="left"))
        k_lo = max(0, min(k_lo, n_bins - 1))
        k_hi = max(k_lo + 1, min(k_hi, n_bins))

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

        plans.append((m_start, m_end, N, hop_g, k_lo, k_hi, starts))
    return plans


# =========================================================================
# 主流程
# =========================================================================
def compute_reassigned_spectrogram(
    samples, sr, midi_min=MIDI_MIN, midi_max=MIDI_MAX,
    rows_per_semitone=ROWS_PER_SEMITONE, target_fps=TARGET_FPS,
    max_frames=65536, progress_cb=None, cancel_cb=None,
):
    n_rows = (midi_max - midi_min + 1) * rows_per_semitone
    samples_np = np.ascontiguousarray(samples, dtype=F32)
    n = len(samples_np)

    if n == 0:
        return np.zeros((1, n_rows), dtype=F32), max(1, sr // target_fps)

    groups_raw = _plan_groups(sr, midi_min, midi_max)

    target_hop = max(1, int(round(sr / float(target_fps))))
    base_hop = max(1, _prev_pow2(target_hop))
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
            w = xp.asarray(np.hanning(N + 1)[:N].astype(F32))
            window_cache[N] = w
        return w

    plans = _plan_chunks(sr, n, groups)
    total_chunks = sum(len(p[6]) for p in plans) or 1
    done_chunks = 0

    if progress_cb is not None:
        try:
            progress_cb(0.0)
        except Exception:
            pass

    cancelled = False
    for m_start, m_end, N, hop_g, k_lo, k_hi, starts in plans:
        if not starts:
            continue

        ratio = int(hop_g // base_hop)
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
                X, N, hop_g, sr, f_k_xp, n_rows, midi_min, midi_max,
                rows_per_semitone, group_grid, i0, k_lo=k_lo, k_hi=k_hi,
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
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
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


def u8_to_qimage(u8, lut):
    rgb_t = np.ascontiguousarray(lut[u8].transpose(1, 0, 2))
    h, w = rgb_t.shape[:2]
    img = QImage(rgb_t.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    return img.copy()


# =========================================================================
# 分析工作线程
# =========================================================================
class AnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    done = pyqtSignal(object, int)
    failed = pyqtSignal(str)

    def __init__(self, samples, sr, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.sr = sr
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def is_cancelled(self):
        return self._cancel

    def run(self):
        try:

            def progress_cb(frac):
                pct = int(max(0.0, min(1.0, float(frac))) * 100)
                self.progress.emit(pct, f"分析中… {pct}%")

            def cancel_cb():
                return self._cancel

            mag, hop = compute_reassigned_spectrogram(
                self.samples, self.sr,
                midi_min=MIDI_MIN, midi_max=MIDI_MAX,
                rows_per_semitone=ROWS_PER_SEMITONE,
                target_fps=TARGET_FPS,
                progress_cb=progress_cb, cancel_cb=cancel_cb,
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
        self.done.emit(db, hop)


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
                return QColor(110, 170, 255) if white_mode else QColor(40, 90, 200)
            if m == self._hover:
                return QColor(196, 218, 255) if white_mode else QColor(66, 92, 148)
            if m in highlight_set:
                return QColor(255, 235, 140) if white_mode else QColor(190, 155, 30)
            return QColor(234, 238, 245) if white_mode else QColor(26, 30, 38)

        for m in range(self.midi_min, self.midi_max + 1):
            if (m % 12) in BLACK_PC:
                continue
            r = self._key_rect(m)
            p.fillRect(r, pick(m, True))
            if m % 12 in (0, 5):
                p.setPen(QPen(QColor(160, 166, 178), 1))
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

            p.setPen(QPen(QColor(160, 166, 178), 1))
            p.drawLine(QPointF(right_x, mid_y), QPointF(r.right(), mid_y))

            left_rect = QRectF(r.x(), r.y(), black_w, r.height())
            p.fillRect(left_rect, pick(m, False))
            p.setPen(QPen(QColor(8, 10, 14), 1))
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
    followModeChanged = pyqtSignal(bool)   # ★ 新增

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
        self.hide_low = 0.0
        self.show_grid = True

        self.harmonics = 0
        self.playhead_frame = None

        self.view_start = 0.0
        self.scale = 1.0

        self.masks = []
        self._mask_drag = None
        self.selected_mask = None

        # ★ 跟随状态
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

    # ------------------------------------------------------------------
    # 跟随控制（对外 API）
    # ------------------------------------------------------------------
    def set_follow_mode(self, on):
        on = bool(on)
        if on == self.follow_mode:
            return
        self.follow_mode = on
        if on:
            # 如果没手动 seek 过 → 使用默认 25%
            if not self._has_manual_seek:
                self._follow_ratio = 0.25
            self._apply_follow()
        self.update()

    def reset_follow_state(self):
        """停止播放时调用：重置比例 + 清手动 seek 标记。"""
        self._follow_ratio = 0.25
        self._has_manual_seek = False

    def _cancel_follow(self):
        """拖动频谱 / 用户干预时调用：关闭跟随 + 重置比例。"""
        changed = False
        if self.follow_mode:
            self.follow_mode = False
            changed = True
        self._follow_ratio = 0.25
        self._has_manual_seek = False
        if changed:
            self.followModeChanged.emit(False)

    def _apply_follow(self):
        """把 view_start 设置为让播放头保持在 _follow_ratio 处。"""
        if self.n_frames <= 0 or self.playhead_frame is None:
            return
        W = max(1, self.width())
        visible = W / self.scale
        if visible >= self.n_frames:
            # 全部可见，无需滚动
            return
        new_view_start = self.playhead_frame - self._follow_ratio * visible
        max_view_start = self.n_frames - visible
        new_view_start = max(0.0, min(new_view_start, max_view_start))
        if abs(new_view_start - self.view_start) * self.scale > 0.5:
            self.view_start = new_view_start
            self._invalidate()

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    def set_data(self, db, hop, sr, fit=True):
        self.db = db
        self.hop = hop
        self.sr = sr
        self.n_frames, self.n_rows = db.shape
        self._regen_u8()
        if self.selected_mask is not None:
            self.selected_mask = None
            self.maskSelected.emit(-1)
        # 换数据后跟随状态重置
        self._has_manual_seek = False
        self._follow_ratio = 0.25
        if fit:
            QTimer.singleShot(0, self.fit_view)
        else:
            self._clamp_view()
            self._invalidate()

    def set_colormap(self, name):
        if name not in LUTS:
            return
        self.cmap = name
        self.lut = LUTS[name]
        if self.u8 is not None:
            self.qimg = u8_to_qimage(self.u8, self.lut)
        self._invalidate()

    def set_hide_low(self, v):
        v = float(v)
        if abs(v - self.hide_low) < 1e-6:
            return
        self.hide_low = v
        if self.db is not None:
            self._regen_u8()
        self._invalidate()

    def set_harmonics(self, n):
        n = max(0, min(5, int(n)))
        if n != self.harmonics:
            self.harmonics = n
            if self._hover is not None and self.n_frames > 0:
                self._emit_hover_info(self._hover)
            if self.selected_mask is not None:
                if self.selected_mask >= len(self.masks):
                    self.selected_mask = None
                    self.maskSelected.emit(-1)
            self.update()

    def set_playhead_frame(self, frame):
        if frame is None:
            if self.playhead_frame is not None:
                self.playhead_frame = None
                self.update()
            return
        if self.playhead_frame is not None and abs(frame - self.playhead_frame) < 0.01:
            return
        self.playhead_frame = float(frame)

        if self.follow_mode and self.n_frames > 0:
            self._apply_follow()
        self.update()

    def set_bpm(self, bpm):
        bpm = float(bpm)
        if bpm <= 0 or abs(bpm - self.bpm) < 1e-9:
            return
        self.bpm = bpm
        self.update()

    def set_beats_per_bar(self, n):
        n = max(1, int(n))
        if n != self.beats_per_bar:
            self.beats_per_bar = n
            self.update()

    def set_show_beats(self, on):
        on = bool(on)
        if on != self.show_beats:
            self.show_beats = on
            self.update()

    def set_beat_offset_frames(self, frames):
        frames = float(frames)
        if abs(frames - self.beat_offset_frames) < 1e-9:
            return
        self.beat_offset_frames = frames
        self.update()

    def clear_masks(self):
        if self.masks or self._mask_drag is not None or self.selected_mask is not None:
            self.masks.clear()
            self._mask_drag = None
            self.selected_mask = None
            self.maskSelected.emit(-1)
            self.update()

    # ------------------------------------------------------------------
    # 遮罩
    # ------------------------------------------------------------------
    def _harmonic_notes(self, midi_n):
        notes = [int(round(midi_n))]
        for k in range(2, self.harmonics + 2):
            mf = midi_n + 12.0 * math.log2(k)
            if self.midi_min - 0.5 <= mf <= self.midi_max + 0.5:
                notes.append(int(round(mf)))
        seen = set()
        result = []
        for n in notes:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result

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

    def _regen_u8(self):
        if self.db is None:
            self.u8 = None
            self.qimg = None
            return
        self.u8 = db_to_u8(self.db, floor_db=DB_FLOOR, hide_low=self.hide_low, gamma=DISPLAY_GAMMA)
        self.qimg = u8_to_qimage(self.u8, self.lut)

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

    def _invalidate(self):
        self._cache = None
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        W = max(1, self.width())
        if self.n_frames > 0:
            ms = W / float(self.n_frames)
            if self.scale < ms:
                self.scale = ms
        self._clamp_view()
        # 若跟随中，重新应用比例（宽度变了 visible 也变）
        if self.follow_mode:
            self._apply_follow()
        self._invalidate()

    def _render_cache(self):
        W, H = max(1, self.width()), max(1, self.height())
        pm = QPixmap(W, H)
        pm.fill(QColor(7, 9, 14))
        if self.qimg is not None and self.n_frames > 0:
            p = QPainter(pm)
            p.setRenderHint(QPainter.SmoothPixmapTransform, self.scale <= 4.0)
            f0 = self.view_start
            f1 = self.view_start + W / self.scale
            sx0 = max(0.0, f0)
            sx1 = min(float(self.n_frames), f1)
            if sx1 > sx0:
                dx0 = (sx0 - f0) * self.scale
                dx1 = (sx1 - f0) * self.scale
                src = QRectF(sx0, 0.0, sx1 - sx0, float(self.n_rows))
                dst = QRectF(dx0, 0.0, dx1 - dx0, float(H))
                p.drawImage(dst, self.qimg, src)
            p.end()
        self._cache = pm

    def _draw_harmonic_marks(self, p, midi_n, W, H):
        marks = [(midi_n, True)]
        for k in range(2, self.harmonics + 2):
            mf = midi_n + 12.0 * math.log2(k)
            if self.midi_min - 1.0 <= mf <= self.midi_max + 1.0:
                marks.append((mf, False))

        for mf, is_fund in marks:
            y_top = midi_to_y(mf + 0.5, H, self.midi_min, self.midi_max)
            y_bot = midi_to_y(mf - 0.5, H, self.midi_min, self.midi_max)
            y_a = max(0.0, min(float(H), y_top))
            y_b = max(0.0, min(float(H), y_bot))
            if y_b <= y_a:
                continue
            if is_fund:
                fill_col = QColor(255, 255, 255, 130)
                line_col = QColor(255, 255, 255, 220)
            else:
                fill_col = QColor(150, 210, 255, 95)
                line_col = QColor(150, 210, 255, 185)
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

            txt = f" {midi_name(midi_n)}   {f_hz:8.2f} Hz   {t:7.3f} s{beat_info} "
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
            self.hoverNote.emit(midi_n)
            self.hoverNotes.emit(self._harmonic_notes(midi_n))

            frame = self.view_start + pos.x() / self.scale
            t = frame * self.hop / float(self.sr)
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
            self._mask_drag = [f, f, midi_n]
            self.setCursor(Qt.SizeHorCursor)
            self.update()
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
                self.update()
                return

            if self.selected_mask is not None:
                self.selected_mask = None
                self.maskSelected.emit(-1)
                self.update()

            midi_n = self._midi_at_pos(e.pos())
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
                self.update()
            self._emit_hover_info(e.pos())
            return

        if self._drag_x0 is not None and self._press_pos is not None:
            dx = e.pos().x() - self._drag_x0
            dy = e.pos().y() - self._press_pos.y()
            if not self._drag_moved and (abs(dx) > 4 or abs(dy) > 4):
                self._drag_moved = True
                # ★ 用户开始拖动平移 → 取消跟随
                self._cancel_follow()
            if self._drag_moved:
                self.view_start = self._drag_view0 - dx / self.scale
                self._clamp_view()
                self._invalidate()

        self.update()
        self._emit_hover_info(e.pos())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton and self._mask_drag is not None:
            f_start, f_end, midi_n = self._mask_drag
            if f_end - f_start < 1.0:
                f_end = min(f_start + 5.0, float(self.n_frames))
            if f_end > f_start:
                self.masks.append((f_start, f_end, midi_n))
                self.selected_mask = len(self.masks) - 1
                self.maskSelected.emit(self.selected_mask)
            self._mask_drag = None
            self.setCursor(Qt.CrossCursor)
            self.update()
            return

        if e.button() == Qt.LeftButton:
            if not self._drag_moved and self._press_pos is not None and self.n_frames > 0:
                # ★ 手动指定播放头位置 → 记录比例，供后续跟随使用
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
        self.update()

    def keyPressEvent(self, e):
        key = e.key()

        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_mask is not None and 0 <= self.selected_mask < len(self.masks):
                del self.masks[self.selected_mask]
                self.selected_mask = None
                self.maskSelected.emit(-1)
                self.update()
                e.accept()
                return
            e.accept()
            return

        if key == Qt.Key_Escape:
            if self.selected_mask is not None:
                self.selected_mask = None
                self.maskSelected.emit(-1)
                self.update()
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

        self._playhead_timer = QTimer(self)
        self._playhead_timer.setInterval(10)
        self._playhead_timer.timeout.connect(self._tick_playhead)

        self.player = None
        self._init_media_player()

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()

        QTimer.singleShot(0, lambda: self.resize(1400, 860))

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

    # -----------------------------------------------------------------
    # 媒体加载
    # -----------------------------------------------------------------
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
            "padding:3px 6px;spacing:4px;}"
            "QToolBar QWidget{background:transparent;}"
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

        act_fit = QAction("适应 🔍", self)
        act_fit.setToolTip("适应窗口 (Ctrl+0)")
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(lambda: self.spec.fit_view())
        tb.addAction(act_fit)

        tb.addSeparator()

        self.act_play = QAction("播放 ▶", self)
        self.act_play.setToolTip("播放 (Space)")
        self.act_play.setShortcut("Space")
        self.act_play.setShortcutContext(Qt.ApplicationShortcut)
        self.act_play.triggered.connect(self._on_play)
        tb.addAction(self.act_play)

        self.act_pause = QAction("暂停 ⏸", self)
        self.act_pause.setToolTip("暂停 (Space)")
        self.act_pause.triggered.connect(self._on_pause)
        tb.addAction(self.act_pause)

        self.act_stop = QAction("停止 ⏹", self)
        self.act_stop.setToolTip("停止")
        self.act_stop.triggered.connect(self._on_stop)
        tb.addAction(self.act_stop)

        if self.player is None:
            self.act_play.setEnabled(False)
            self.act_pause.setEnabled(False)
            self.act_stop.setEnabled(False)

        tb.addSeparator()

        self.act_follow = QAction("跟随 🎯", self)
        self.act_follow.setCheckable(True)
        self.act_follow.setChecked(False)
        self.act_follow.setToolTip("屏幕跟随播放头（拖动频谱自动取消）")
        self.act_follow.toggled.connect(self._on_follow_toggled)
        tb.addAction(self.act_follow)

        tb.addSeparator()

        tb.addWidget(QLabel("主题"))
        self.cb_cmap = QComboBox()
        self.cb_cmap.setToolTip("颜色映射")
        self.cb_cmap.setFixedWidth(75)
        for name in ("magma", "inferno", "viridis", "ice"):
            self.cb_cmap.addItem(name, name)
        self.cb_cmap.currentIndexChanged.connect(lambda i: self.spec.set_colormap(self.cb_cmap.itemData(i)))
        tb.addWidget(self.cb_cmap)

        tb.addWidget(QLabel("泛音数量"))
        self.cb_harm = QComboBox()
        self.cb_harm.setToolTip("高亮的泛音数量（0 = 只基音）")
        self.cb_harm.setFixedWidth(40)
        for n in range(6):
            self.cb_harm.addItem(str(n), n)
        self.cb_harm.currentIndexChanged.connect(lambda i: self.spec.set_harmonics(self.cb_harm.itemData(i)))
        tb.addWidget(self.cb_harm)

        tb.addSeparator()

        self.act_show_beats = QAction("节拍线 §", self)
        self.act_show_beats.setCheckable(True)
        self.act_show_beats.setChecked(True)
        self.act_show_beats.setToolTip("显示/隐藏节拍线与小节线")
        self.act_show_beats.toggled.connect(self._on_show_beats_toggled)
        tb.addAction(self.act_show_beats)

        tb.addWidget(QLabel("BPM"))
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(20.0, 400.0)
        self.spin_bpm.setDecimals(2)
        self.spin_bpm.setSingleStep(1.0)
        self.spin_bpm.setValue(120.0)
        self.spin_bpm.setFixedWidth(60)
        self.spin_bpm.setToolTip("每分钟拍数，用于绘制节拍线")
        self.spin_bpm.valueChanged.connect(self._on_bpm_changed)
        tb.addWidget(self.spin_bpm)

        tb.addWidget(QLabel("拍/小节"))
        self.spin_bpb = QSpinBox()
        self.spin_bpb.setRange(1, 16)
        self.spin_bpb.setValue(4)
        self.spin_bpb.setFixedWidth(30)
        self.spin_bpb.setToolTip("每小节拍数（小节线位置）")
        self.spin_bpb.valueChanged.connect(lambda v: self.spec.set_beats_per_bar(v))
        tb.addWidget(self.spin_bpb)

        tb.addSeparator()

        self.act_clear_mask = QAction("清空遮罩 🧹", self)
        self.act_clear_mask.setToolTip("清除所有遮罩")
        self.act_clear_mask.triggered.connect(self._on_clear_masks)
        tb.addAction(self.act_clear_mask)

        tb.addSeparator()

        tb.addWidget(QLabel("滤镜"))
        self.sld_hide = QSlider(Qt.Horizontal)
        self.sld_hide.setRange(0, 100)
        self.sld_hide.setValue(35)
        self.sld_hide.setFixedWidth(130)
        self.sld_hide.valueChanged.connect(self._on_hide_slider_changed)
        tb.addWidget(self.sld_hide)

        self.lbl_hide = QLabel("35%")
        self.lbl_hide.setStyleSheet("color:#9fc5ff;font-family:Consolas,Menlo,monospace;" "min-width:34px;background:transparent;")
        tb.addWidget(self.lbl_hide)
        self.hide_low = self.sld_hide.value() / 100.0
        self.spec.set_hide_low(self.hide_low)

    def _build_statusbar(self):
        sb = QStatusBar()
        sb.setStyleSheet("QStatusBar{background:#000000;color:#cfd8ea;" "border-top:1px solid #262c3a;}" "QStatusBar::item{border:none;}" "QStatusBar QLabel{color:#cfd8ea;background:transparent;}")
        self.setStatusBar(sb)

        self.lbl_left = QLabel("就绪  ·  拖入音频文件或点击「打开音频」")
        self.lbl_left.setStyleSheet("color:#cfd8ea;background:transparent;")
        self.lbl_center = QLabel("")
        midi_txt = f"MIDI: {MIDI_PORT_NAME}" if MIDI_AVAILABLE else "MIDI: 不可用"
        self.lbl_right = QLabel(f"{GPU_NAME}  ·  {midi_txt}  ·  " f"滚轮缩放 · 左键平移/发声 · 右键拖动创建遮罩")
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
        self.btn_cancel.clicked.connect(self._cancel_analysis)
        sb.addPermanentWidget(self.btn_cancel, 0)

        sb.addPermanentWidget(self.lbl_center, 0)
        sb.addPermanentWidget(self.lbl_right, 0)

    # -----------------------------------------------------------------
    # 事件响应
    # -----------------------------------------------------------------
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

    # ★ 跟随模式
    def _on_follow_toggled(self, on):
        self.spec.set_follow_mode(bool(on))
        if on:
            self.lbl_left.setText("跟随模式已开启")
        else:
            self.lbl_left.setText("跟随模式已关闭")

    def _on_follow_changed_externally(self, on):
        # 来自 SpectrogramView（拖动取消）
        if self.act_follow.isChecked() != on:
            self.act_follow.setChecked(on)
        self.lbl_left.setText("跟随模式已开启" if on else "跟随模式已关闭（拖动已取消）")

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
            self.lbl_left.setText(
                f"跳转至  {frame_pos * self.hop / self.sr:.3f} s   ·   "
                f"{midi_name(midi_note)}"
            )

    def _play_midi_note(self, midi):
        if not MIDI_AVAILABLE:
            return
        if self._active_note is not None:
            midi_note_off(self._active_note)
        midi_note_on(midi, vel=90)
        self._active_note = midi
        self._note_off_timer.start()

    def _stop_active_note(self):
        if self._active_note is not None:
            midi_note_off(self._active_note)
            self._active_note = None

    def _on_hide_slider_changed(self, v):
        self.lbl_hide.setText(f"{v}%")
        self._hide_timer.start()

    def _apply_hide_low(self):
        v = self.sld_hide.value()
        new_val = v / 100.0
        if abs(new_val - self.hide_low) < 1e-6:
            return
        self.hide_low = new_val
        self.spec.set_hide_low(self.hide_low)

    def _on_clear_masks(self):
        self.spec.clear_masks()
        self.lbl_left.setText("已清除所有遮罩")

    def _on_bpm_changed(self, v):
        self.spec.set_bpm(float(v))
        self.lbl_left.setText(f"BPM = {float(v):.2f}  ·  拍/小节 = {self.spin_bpb.value()}")

    def _on_show_beats_toggled(self, on):
        self.spec.set_show_beats(bool(on))
        self.lbl_left.setText("已显示节拍线" if on else "已隐藏节拍线")

    # -----------------------------------------------------------------
    # 播放控制
    # -----------------------------------------------------------------
    def _on_play(self):
        if self.player is None or self.current_path is None:
            return
        if self._is_playing:
            self._on_pause()
            return
        try:
            self._set_player_volume(1.0)
            self.player.play()
        except Exception as e:
            print(f"[media] play 失败: {e}")
            return
        self._warming = False
        self._is_playing = True
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
        self._playhead_timer.stop()
        # ★ 停止 → 重置跟随比例与手动 seek 标记，之后播放头从起点自然起步
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
        except Exception:
            pass

    # -----------------------------------------------------------------
    # 打开 / 加载
    # -----------------------------------------------------------------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开音频文件", "", "音频文件 (*.wav *.flac *.ogg *.mp3 *.m4a *.aiff *.aif);;所有文件 (*)")
        if path:
            self.load_path(path)

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
        self.lbl_left.setText(f"{name}   ·   {sr} Hz   ·   {dur:.2f} s{ch_txt}   ·   正在分析…")

        self._rebuild(fit=True)

    # -----------------------------------------------------------------
    # 异步分析
    # -----------------------------------------------------------------
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
            self.act_pause.setEnabled(False)
            self.act_stop.setEnabled(False)
        else:
            self.progress_bar.setVisible(False)
            self.btn_cancel.setVisible(False)
            if self.player is not None:
                self.act_play.setEnabled(True)
                self.act_pause.setEnabled(True)
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

    def _on_analysis_done(self, db, hop):
        self._set_busy(False)

        self.hop = hop
        self.spec.hop = hop
        self.spec.sr = self.sr
        self.spec.set_data(db, hop, self.sr, fit=True)
        self.spec.set_playhead_frame(0.0)

        n_frames, n_rows = db.shape
        backend = "GPU" if HAS_GPU else "CPU"
        self.lbl_left.setText(f"就绪  ·  {backend}  ·  " f"{n_frames} 帧 × {n_rows} 行  ·  hop {hop} " f"({self.sr / hop:.1f} fps)")
        self.lbl_center.setText("")
        self.spec.setFocus()

    def _rebuild(self, fit=True):
        if self.samples is None:
            return

        self._cancel_analysis(wait=True)

        backend = "GPU" if HAS_GPU else "CPU"
        self.progress_bar.setValue(0)
        self._set_busy(True)
        self.lbl_left.setText(f"多分辨率谱重分配 (三帧相位 + 自适应 sigma) · {backend} 计算中…")

        worker = AnalysisWorker(self.samples, self.sr, self)
        worker.progress.connect(self._on_analysis_progress)
        worker.done.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))

        self._worker = worker
        worker.start()

    def _on_worker_finished(self, w):
        if self._worker is w:
            self._worker = None

    # -----------------------------------------------------------------
    # 拖拽 / 关闭
    # -----------------------------------------------------------------
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
                "positionChanged", "stateChanged", "mediaStatusChanged",
                "error", "durationChanged",
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

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    app.setStyleSheet(
        "QWidget{background:#000000;color:#cfd8ea;}"
        "QMainWindow{background:#000000;}"
        "QToolTip{background:#000000;color:#cfd8ea;"
        "border:1px solid #303848;padding:3px 6px;}"
        "QMenu{background:#000000;color:#cfd8ea;border:1px solid #303848;}"
        "QMenu::item:selected{background:#2b5fb8;color:#ffffff;}"
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