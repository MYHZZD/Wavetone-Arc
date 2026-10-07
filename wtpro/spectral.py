"""频谱分析：窗函数、多分辨率分组、STFT、相位重分配、多群组合成。

本层只产出线性幅度，不涉及 dB、配色或显示域。"""

import math
import os
import time

import numpy as np

from wtpro import backends
from wtpro.backends import XP_LOCK, _ensure_backend_ready
from wtpro.common import (
    DB_FLOOR, DEFAULT_ROWS_PER_SEMITONE, DEFAULT_TARGET_FPS, DEFAULT_WINDOW,
    ENERGY_THRESHOLD_DB, F32, F64, FFT_MAX, FFT_MIN, GROUP_OVERLAP, GROUP_SEMITONES,
    MIDI_MAX, MIDI_MIN, SPLAT_K, SPLAT_SIGMA_BINS, SPLAT_SIGMA_MIN_ROWS,
    TIME_INV_SIGMA_SQ_HALF, TIME_SPLAT_K, _int_opt, _midpoint_freq, _next_pow2,
    _prev_pow2, midi_name, midi_to_freq
)


def _xp():
    """返回当前后端对应的数组模块（numpy 或 cupy）。

    backends.xp 在分析前后可能被 _activate_backend 改写，
    所以不能在这里做成模块级常量，必须每次现取。
    """
    return backends.xp


def _sync():
    """同步当前后端。CPU 上是空操作；GPU 上必须有它，否则计时只量到下发。

    计时不准比没有计时更糟 —— 只有下发时间的话，会得出"GPU 比 CPU 快百倍"
    这种荒谬结论。
    """
    xp = backends.xp
    if xp is np:
        return
    try:
        xp.cuda.Stream.null.synchronize()
    except Exception:
        pass

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

        target_mem = 1.5e8 if backends.HAS_GPU else 4.0e8
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
# 定义在本层（而不是 worker）是因为抛它的是 _compute_reassigned_locked。
# 之前只在 worker 里定义了同名类，于是用户点"取消"时这里抛的是
# NameError 而不是取消异常 —— 取消会变成一次"分析失败"弹窗。


class AnalysisCancelled(Exception):
    pass


def _stft_batch(samples_xp, N, hop, i0, i1, window_xp):
    n = int(samples_xp.shape[0])
    half = N // 2

    starts = _xp().arange(i0, i1, dtype=_xp().int64) * hop - half
    base = _xp().arange(N, dtype=_xp().int64)
    idx = starts[:, None] + base[None, :]

    valid = (idx >= 0) & (idx < n)
    _xp().clip(idx, 0, n - 1, out=idx)

    frames = samples_xp[idx]
    frames *= valid
    frames *= window_xp[None, :]

    X = _xp().fft.rfft(frames, axis=1)
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
    interp=True,
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

    mag = _xp().abs(X_sub)
    phase = _xp().angle(X_sub)

    # 三帧相位差 → 瞬时频率（频率方向的重分配）
    dphase = (phase[2:] - phase[:-2]) * 0.5
    mag_mid = mag[1:-1]
    del mag
    del phase

    k_idx = _xp().arange(k_lo, k_hi, dtype=_xp().float64)
    # 减掉每个 bin 在 hop 之间的固有相位推进，再折算成频率偏差。
    # 折到 (-pi, pi] 的写法：((x + pi) mod 2pi) - pi。
    dphase -= (2.0 * _xp().pi * float(hop) / float(N)) * k_idx[None, :]
    dphase = _xp().mod(dphase + _xp().pi, 2.0 * _xp().pi) - _xp().pi
    dphase *= float(sr) / (2.0 * _xp().pi * float(hop))

    f_inst = f_k_sub[None, :] + dphase
    del dphase
    f_inst = _xp().maximum(f_inst, F64(1e-9))

    m_midi = 69.0 + 12.0 * _xp().log2(f_inst * (1.0 / 440.0))
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
        frame_max = _xp().maximum(frame_max, F32(1e-12))
        threshold = frame_max * F32(10.0 ** (ENERGY_THRESHOLD_DB / 20.0))
        valid = valid & (mag_mid > threshold)
        del frame_max, threshold

    tv, kv = _xp().where(valid)
    del valid
    if tv.size == 0:
        return

    # 不插值：每个 bin 直接落到最近的一行，不做任何撒点。
    # 画面会锐利但出现横向梳状条纹（相邻 bin 之间有空隙），用于对照。
    if not interp:
        r_near = _xp().round(r_float[tv, kv]).astype(_xp().int64)
        ok = (r_near >= 0) & (r_near < n_rows)
        if not bool(ok.any()):
            return
        global_t = tv + 1
        t_lo = int(global_t.min())
        t_hi = int(global_t.max()) + 1
        span = t_hi - t_lo
        if span <= 0:
            return
        flat = (global_t[ok] - t_lo) * n_rows + r_near[ok]
        contrib = _xp().bincount(flat, weights=mag_mid[tv, kv][ok],
                              minlength=span * n_rows).astype(F32)
        group_grid[i0_group + t_lo : i0_group + t_hi] += contrib.reshape(span, n_rows)
        return

    r_v = r_float[tv, kv]
    mag_v = mag_mid[tv, kv]
    del r_float, mag_mid

    # σ 自适应：核宽取 SPLAT_SIGMA_BINS 个 bin 宽，再换算成行。
    # 行宽（1 行对应多少 Hz）= 半音间距 / rows_per_semitone，
    # 半音间距 = f·(2^(1/12)−1)，f 为该行中心频率。
    m_midi_v = (float(midi_max) + 0.5) - (r_v + 0.5) / float(rows_per_semitone)
    f_row_v = 440.0 * _xp().power(2.0, (m_midi_v - 69.0) / 12.0)
    del m_midi_v
    row_hz_v = f_row_v * ((2.0 ** (1.0 / 12.0) - 1.0) / float(rows_per_semitone))
    del f_row_v
    bin_hz = float(sr) / float(N)
    sigma_rows_v = SPLAT_SIGMA_BINS * bin_hz / _xp().maximum(row_hz_v, 1e-12)
    sigma_rows_v = _xp().maximum(sigma_rows_v, SPLAT_SIGMA_MIN_ROWS).astype(F32)
    del row_hz_v
    inv_sigma_sq_half_v = (-0.5 / (sigma_rows_v * sigma_rows_v)).astype(F32)
    del sigma_rows_v

    r_base = _xp().floor(r_v).astype(_xp().int64)
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

    # 归一化因子：该 bin 撒到全部 (2·SPLAT_K+1) 个候选行上的权重之和。
    # σ 逐点不同，所以这个和也必须逐点算，不能全局取一个常数。
    # 放在循环外只算一次，避免每个 off 都重算一遍。
    w_tot = _xp().zeros_like(frac)
    for off in range(-SPLAT_K, SPLAT_K + 1):
        d = frac - off
        w_tot += _xp().exp(inv_sigma_sq_half_v * d * d)
        del d
    _xp().maximum(w_tot, F32(1e-20), out=w_tot)

    for off in range(-SPLAT_K, SPLAT_K + 1):
        r_target = r_base + off
        d = frac - off
        w = _xp().exp(inv_sigma_sq_half_v * d * d) / w_tot
        del d

        ok = (r_target >= 0) & (r_target < n_rows)
        if not bool(ok.any()):
            continue

        t_ok = t_shift[ok]
        r_ok = r_target[ok]
        v_ok = mag_v[ok] * w[ok]
        flat = t_ok * n_rows + r_ok

        contrib = _xp().bincount(flat, weights=v_ok, minlength=size).astype(F32)
        group_grid[i0_group + t_lo : i0_group + t_hi] += contrib.reshape(span, n_rows)

    # 归一化因子：该 bin 撒到全部 (2·SPLAT_K+1) 个候选行上的权重之和。


# =========================================================================
# 群组网格 → 公共网格：时间方向的高斯重采样
# =========================================================================


def _splat_group_to_common(group_grid, ratio, out, n_common, interp=True):
    n_g, n_rows = group_grid.shape
    if n_g <= 0:
        return

    if ratio == 1:
        m = min(n_g, n_common)
        if m > 0:
            out[:m] += group_grid[:m]
        return

    t_arr = _xp().arange(n_common, dtype=_xp().float64) / float(ratio)
    i0 = _xp().floor(t_arr).astype(_xp().int64)
    frac = (t_arr - i0).astype(F32)

    # 不插值：每个输出帧直接取最近的组帧。
    # 各频段的 hop 相差最多 128 倍，这样会在时间方向出现宽为 ratio 帧的台阶。
    if not interp:
        i_near = _xp().clip(_xp().round(t_arr).astype(_xp().int64), 0, n_g - 1)
        out += group_grid[i_near]
        return

    # 归一化因子：核在 ±TIME_SPLAT_K 处被截断，不归一化时权重和约 1.5
    # （相当于整体放大 1.5 倍）。除以它，时间插值才是真正的加权平均。
    w_tot = _xp().zeros_like(frac)
    for off in range(-TIME_SPLAT_K, TIME_SPLAT_K + 1):
        d = frac - off
        w_tot += _xp().exp(TIME_INV_SIGMA_SQ_HALF * d * d)
    _xp().maximum(w_tot, F32(1e-20), out=w_tot)

    for off in range(-TIME_SPLAT_K, TIME_SPLAT_K + 1):
        idx = i0 + off
        valid = (idx >= 0) & (idx < n_g)
        if not bool(valid.any()):
            continue

        d = frac - off
        w = _xp().exp(TIME_INV_SIGMA_SQ_HALF * d * d).astype(F32) / w_tot
        w = w * valid
        if not bool((w > 1e-6).any()):
            continue

        idx_clipped = _xp().clip(idx, 0, n_g - 1)
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
    interp=True,
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
            interp=interp,
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
    interp=True,
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

    samples_xp = _xp().asarray(samples_np)
    out_xp = _xp().zeros((n_common, n_rows), dtype=F32)

    window_cache = {}

    def get_window(N):
        w = window_cache.get(N)
        if w is None:
            w = _xp().asarray(make_window(window_name, N))
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
            print(f"[analyze] 信号过短（{n} 个采样 ≈ {n / float(sr) * 1000:.1f} ms）："
                  f"最长窗可得的帧数仅为 {min_frames}，不足 3 帧。谱重分配需要连续"
                  f"三帧计算瞬时频率，故本次分析结果为空。")
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
                print(f"[analyze] 信号偏短（{n / float(sr):.2f} s）：最长 FFT 受信号长度"
                      f"限制，{midi_name(ms)} 附近的分辨率为 {df:.1f} Hz，已达半音间距的"
                      f"{ratio:.1f} 倍，该频段内相邻半音无法分离。")

    if progress_cb is not None:
        try:
            progress_cb(0.0)
        except Exception:
            pass

    cancelled = False
    # 分阶段计时（只在 WAVETONEPRO_TIMING=1 时打印）。
    # 为什么需要它：CPU 与 GPU 的"贵"在哪一段完全不同 ——
    # GPU 上 FFT 很快，但**每组的 _splat_group_to_common 与反复的小 kernel
    # 下发**会成为瓶颈；CPU 上则是 FFT 本身。没有分阶段的数字就只能猜。
    _timing = os.environ.get("WAVETONEPRO_TIMING", "") not in ("", "0")
    _t_stft = _t_reassign = _t_splat = 0.0
    if _timing:
        print(f"[timing] 后端 {backends.GPU_NAME}  音频 {n} 样本"
              f"（{n/float(sr):.2f}s）  组数 {len(plans)}  分块 {total_chunks}  "
              f"公共帧 {n_common}  base_hop {base_hop}  行 {n_rows}", flush=True)
    for gi, (m_start, m_end, N, hop_g, k_lo, k_hi, starts, band_lo, band_hi) \
            in enumerate(plans):
        if not starts:
            continue

        ratio = max(1, int(hop_g // base_hop))
        n_g_frames = n // hop_g + 1
        if _timing:
            print(f"[timing] 组 {gi+1}/{len(plans)}  MIDI {midi_name(m_start)}"
                  f"–{midi_name(m_end)}  N={N}  hop={hop_g}  ratio={ratio}  "
                  f"组帧 {n_g_frames}  分块 {len(starts)}", flush=True)

        group_grid = _xp().zeros((n_g_frames, n_rows), dtype=F32)
        window_xp = get_window(N)
        n_bins = N // 2 + 1
        f_k_xp = _xp().arange(n_bins, dtype=_xp().float64) * (float(sr) / float(N))

        for ci, (i0, i1) in enumerate(starts):
            if cancel_cb is not None:
                try:
                    if cancel_cb():
                        cancelled = True
                        break
                except Exception:
                    pass

            _t0 = time.perf_counter()
            X = _stft_batch(samples_xp, N, hop_g, i0, i1, window_xp)
            if _timing:
                _sync()
                _d = time.perf_counter() - _t0
                _t_stft += _d
                print(f"[timing]   STFT      块 {ci+1}/{len(starts)}  "
                      f"{_d:8.3f}s  X{X.shape}", flush=True)
            _t0 = time.perf_counter()
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
                interp=interp,
            )
            if _timing:
                _sync()
                _d = time.perf_counter() - _t0
                _t_reassign += _d
                print(f"[timing]   重分配    块 {ci+1}/{len(starts)}  "
                      f"{_d:8.3f}s", flush=True)
            del X

            done_chunks += 1
            if progress_cb is not None:
                try:
                    progress_cb(done_chunks / total_chunks)
                except Exception:
                    pass

        if cancelled:
            break

        _t0 = time.perf_counter()
        _splat_group_to_common(group_grid, ratio, out_xp, n_common,
                               interp=interp)
        if _timing:
            _sync()
            _d = time.perf_counter() - _t0
            _t_splat += _d
            print(f"[timing]   组间插值  {_d:8.3f}s  ratio={ratio}", flush=True)
        del group_grid

    if cancelled:
        raise AnalysisCancelled()

    if _timing:
        _t0 = time.perf_counter()
        if backends.HAS_GPU:
            out_np = _xp().asnumpy(out_xp).astype(F32, copy=False)
        else:
            out_np = np.asarray(out_xp, dtype=F32)
        _t_copy = time.perf_counter() - _t0
        _total = _t_stft + _t_reassign + _t_splat + _t_copy
        print(f"[timing] 后端 {backends.GPU_NAME}  "
              f"音频 {n} 样本（{n/float(sr):.2f}s）  输出 {out_np.shape}")
        print(f"[timing]   STFT        {_t_stft:8.3f}s  {100*_t_stft/max(_total,1e-9):5.1f}%")
        print(f"[timing]   重分配      {_t_reassign:8.3f}s  {100*_t_reassign/max(_total,1e-9):5.1f}%")
        print(f"[timing]   组间插值    {_t_splat:8.3f}s  {100*_t_splat/max(_total,1e-9):5.1f}%")
        print(f"[timing]   回传 CPU    {_t_copy:8.3f}s  {100*_t_copy/max(_total,1e-9):5.1f}%")
        print(f"[timing]   合计        {_total:8.3f}s"
              f"  ({n/float(sr)/max(_total,1e-9):.0f} 倍实时)")
        print(f"[timing]   组数 {len(plans)}  分块 {total_chunks}  "
              f"公共帧 {n_common}  base_hop {base_hop}")
        return out_np, base_hop

    if backends.HAS_GPU:
        out_np = _xp().asnumpy(out_xp).astype(F32, copy=False)
    else:
        out_np = np.asarray(out_xp, dtype=F32)

    return out_np, base_hop


# =========================================================================
# dB / 显示
# =========================================================================


def compute_db(mag, floor_db=DB_FLOOR, ref=None):
    """线性幅度 → 相对 dB（上限 0，下限 floor_db）。

    ref 是归一的参考幅度。默认 None 表示用**本矩阵自己的最大值**，即原有的
    "按整幅峰值归一到 0 dB"。

    多平面（L / R / stereo）时必须传入**所有平面共用的**一个参考值：
    否则每个平面各除以各的峰值，左右声道之间的真实电平差会被抹平，
    三张图也没法放在一起比较。调用方直接传 max(各平面 mag.max()) 即可。
    """
    if mag.size == 0:
        return mag
    if ref is None:
        mx = float(mag.max())
    else:
        try:
            mx = float(ref)
        except (TypeError, ValueError):
            mx = float(mag.max())
    if not np.isfinite(mx) or mx <= 1e-20:
        return np.full(mag.shape, floor_db, dtype=F32)

    db = np.empty_like(mag, dtype=F32)
    np.maximum(mag, F32(1e-12), out=db)
    np.log10(db, out=db)
    db *= F32(20.0)
    db -= F32(20.0 * math.log10(mx))
    np.clip(db, floor_db, 0.0, out=db)
    return db


