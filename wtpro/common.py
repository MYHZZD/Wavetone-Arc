"""公共常量与坐标换算：音高/MIDI/频率、和弦定义、显示与分析的默认参数。

本模块不依赖其它 wtpro 子模块，是整包依赖树的根。"""

import math

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


MIDI_MIN = 21


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


BLACK_PC = {1, 3, 6, 8, 10}


MIDI_MAX = 108


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


CUSTOM_CHORD_NAME = "自定义"


CHORD_NONE = "无"


CUSTOM_CHORD_LO = 60                 # 自定义和弦窗的两个八度: C4..B5


CUSTOM_CHORD_HI = 83




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




def _next_pow2(n):
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()




def _prev_pow2(n):
    n = max(1, int(n))
    return 1 << (n.bit_length() - 1)




def _midpoint_freq(f_a, f_b):
    """两个频率的几何中点（音乐上是它们之间的半音中点）。"""
    return math.sqrt(float(f_a) * float(f_b))




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
# 频率方向高斯撒点的核宽，以"bin 宽度"为单位自适应：
#     sigma_Hz = SPLAT_SIGMA_BINS × (sr / N)
# 换算成行：sigma_rows = sigma_Hz / (该行的 1 行对应多少 Hz)
#
# 为什么按 bin 宽而不是固定行数：
#   * bin 宽 = 该频段真正能分辨的频率间距，是物理分辨率的上限；
#   * 一个半音占 rows_per_semitone 行（可调），所以"1 行 = 多少 Hz"随行数和音高变化；
#   * 若 σ 固定成行数，窄 bin 的频段就会在相邻 bin 之间留下覆盖缝隙，
#     行网格落进缝隙的地方拿不到能量 —— 表现为横向的明暗条纹（梳状纹）。
#   * 取 σ 正比于 bin 宽，则相邻 bin 的高斯覆盖始终有足够重叠，条纹消失，
#     且"抹开的宽度"永远等于物理分辨率，不会过度模糊。
# 系数 2.0 由真实音频标定：相邻行亮度差的中位数从 3.25 dB 降到 1.29 dB，
# 超过 6 dB 的相邻行对从 23% 降到 2%；再往上加收益饱和并开始糊掉真实频谱。


SPLAT_SIGMA_BINS = 2.0


SPLAT_SIGMA_MIN_ROWS = 0.35      # 下限：防止极窄 bin 时核退化成单行



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


# =========================================================================
# 声道选择
# -------------------------------------------------------------------------
# 分析模式由「分析参数」窗口里的 L / R 两个复选框决定，共四种：
#
#     stereo_only   都不勾   只算左右平均，出 1 张图
#     l_only        勾 L     只算左声道，出 1 张图
#     r_only        勾 R     只算右声道，出 1 张图
#     lr_both       都勾     算 L 与 R，再平均出 stereo，共 3 张图
#
# **最多只算 2 次分析**（lr_both）：stereo 由 (|L| + |R|) / 2 后处理得出，
# 不必再单独分析一遍。这里刻意**不做代数求和**——把两路样本先加起来再分析
# 会让反相内容互相抵消（整段消失），幅度平均则不会。
#
# 显示哪一张由「谱面设置 - 外观 - 声道」下拉框决定，与分析参数完全解耦：
# 算出哪些平面是分析参数的事，看哪一张是显示参数的事。
# =========================================================================


CHANNEL_MODES = ("stereo_only", "l_only", "r_only", "lr_both")


DEFAULT_CHANNEL_MODE = "stereo_only"


# 每种模式实际会算出的平面，顺序即下拉框的候选顺序（stereo → l → r）
MODE_PLANES = {
    "stereo_only": ("stereo",),
    "l_only": ("l",),
    "r_only": ("r",),
    "lr_both": ("stereo", "l", "r"),
}


# 平面名 → 中文标签
PLANE_LABELS = {
    "stereo": "立体声（L/R 平均）",
    "l": "左声道 (L)",
    "r": "右声道 (R)",
}


def channel_mode_from_flags(want_l, want_r):
    """L / R 两个复选框 → 分析模式名。

    都不勾时退回 stereo_only（左右平均），也就是原有行为。
    """
    want_l = bool(want_l)
    want_r = bool(want_r)
    if want_l and want_r:
        return "lr_both"
    if want_l:
        return "l_only"
    if want_r:
        return "r_only"
    return "stereo_only"


def mode_axes(channel_mode):
    """分析模式 → 要处理的声道轴列表（立体声输入里的第几路）。"""
    mode = channel_mode if channel_mode in CHANNEL_MODES else DEFAULT_CHANNEL_MODE
    if mode == "stereo_only":
        return ()
    if mode == "l_only":
        return (0,)
    if mode == "r_only":
        return (1,)
    return (0, 1)


def axis_samples(samples, axis):
    """取立体声输入的第 axis 路，返回 1 维连续 float32。

    单声道输入（1 维，或 2 维但只有 1 列）时两路都返回同一份数据 ——
    "没有右声道"和"右声道等于左声道"在多声道分析上是等价的，
    没必要为它单独维护一个分支。
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        return np.ascontiguousarray(x)
    if x.ndim != 2:
        raise ValueError(f"不支持的采样维度: {x.ndim}")
    if x.shape[1] == 1 or axis >= x.shape[1]:
        return np.ascontiguousarray(x[:, 0])
    return np.ascontiguousarray(x[:, axis])


def stereo_average(samples):
    """立体声折叠：左右两路的算术平均。

    这就是原有 to_mono 的行为，也是原版「双声道求和」的等价形式（差一个
    1/2 的缩放，不影响后续的相对关系）。注意它是**代数**平均 ——
    反相内容会在这里互相抵消，这正是 stereo 平面改用幅度平均的原因。
    """
    return to_mono(samples)


def mean_planes(a, b):
    """两个幅度平面逐元素平均，得到 stereo 平面。

    这就是「stereo = 左右平均」的定义：先各自分析取模，再平均。
    与 stereo_average 的差别只在相位：反相时这里的平均**不会**抵消。
    两个平面来自同一份配置（同 hop、同分组），形状必然一致。
    """
    return (np.asarray(a, dtype=F32) * F32(0.5)
            + np.asarray(b, dtype=F32) * F32(0.5)).astype(F32, copy=False)


