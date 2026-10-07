"""配色：内置配色表、WaveTone 渐变的解码与重采样、线性域饱和起点的换算。"""

import base64
import math

import numpy as np


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
    """把色标线性插值成 n 级 LUT。

    注意这里**不**强制 lut[0] = 纯黑：各个配色的首个色标本身就是它的暗端终点，
    viridis 的暗端是 (68,1,84) 的暗紫而非黑色，强行改成黑会削掉它的固有特征
    （实测被替换掉的感知色差 ΔE≈54），在线性域这种把大片低电平压到最低几级的
    映射下会看到一条明显的分界。
    """
    stops = np.asarray(stops, dtype=np.float64) / 255.0
    x = np.linspace(0.0, 1.0, len(stops))
    xi = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(xi, x, stops[:, c]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return lut




LUTS = {
    "magma": build_lut(MAGMA_STOPS),
    "inferno": build_lut(INFERNO_STOPS),
    "viridis": build_lut(VIRIDIS_STOPS),
    "ice": build_lut(ICE_STOPS),
}


LUTS_RGB = {name: (lut[:, 0].copy(), lut[:, 1].copy(), lut[:, 2].copy()) for name, lut in LUTS.items()}


# =========================================================================
# WaveTone 2.74 同款渐变 + 线性显示域
# -------------------------------------------------------------------------
# 渐变数据取自 WaveTone 自带的 data\wtgcolor.bmp（192 宽 × 10 高 × 24bpp）：
#   列 = 亮度级别 0..191，行 = 配色方案（对应 ini 的 graphcol 0..9，BMP 自下而上）。
# 该 BMP 里方案 0 / 1 / 9 内容完全相同，都是正版默认那套（黑 → 青 → 红），
# 其余方案本工具用不上，所以只内嵌这一套共 576 字节，base64 保证自包含。
#
# 注意：渐变**底部不是纯黑**（level0=(0,0,0)、level1=(0,0,4)、level2=(0,0,8)…），
# 这是它暗场层次明显的原因之一。这里保留原始数据，不做 lut[0] 覆写
# （build_lut 同样不再覆写，见其说明）。
# =========================================================================


_WT_PAL_B64 = (
    "AAAAAAAEAAAIAAAMAAAQAAAUAAAYAAAcAAAgAAAkAAAoAAAsAAAwAAA0AAA4AAA8AABAAABEAABIAABMAABQAABUAABYAABcAABg"
    "AABkAABoAABsAABwAAB0AAB4AAB8AACAAACEAACIAACMAACQAACUAACYAACcAACgAACkAACoAACsAACwAAC0AAC4AAC8AADAAADE"
    "AADIAADMAADQAADUAADYAADcAADgAADkAADoAADsAADwAAD0AAD4AAD8AAj/ABD/ABj/ACD/ACj/ADD/ADj/AED/AEj/AFD/AFj/"
    "AGD/AGj/AHD/AHj/AID/AIf/AI//AJf/AJ//AKf/AK//ALf/AL//AMf/AM//ANf/AN//AOf/AO//APf/AP//AP/3AP/vAP/nAP/f"
    "AP/XAP/PAP/HAP+/AP+3AP+vAP+nAP+fAP+XAP+PAP+HAP+AAP94AP9wAP9oAP9gAP9YAP9QAP9IAP9AAP84AP8wAP8oAP8gAP8Y"
    "AP8QAP8IAP8ACP8AEP8AGP8AIP8AKP8AMP8AOP8AQP8ASP8AUP8AWP8AYP8AaP8AcP8AeP8AgP8Ah/8Aj/8Al/8An/8Ap/8Ar/8A"
    "t/8Av/8Ax/8Az/8A1/8A3/8A5/8A7/8A9/8A//8A//cA/+8A/+cA/98A/9cA/88A/8cA/78A/7cA/68A/6cA/58A/5cA/48A/4cA"
    "/4AA/3gA/3AA/2gA/2AA/1gA/1AA/0gA/0AA/zgA/zAA/ygA/yAA/xgA/xAA/wgA/wAA"
)


WT_PALETTE_W = 192




def _wt_palette():
    """解码内置渐变，返回 (192, 3) 的 RGB，下标 0 = 最暗、191 = 最亮。"""
    raw = base64.b64decode(_WT_PAL_B64)
    return np.frombuffer(raw, np.uint8).reshape(WT_PALETTE_W, 3)




def _wt_lut(n=256):
    """把正版的 192 级渐变重采样到 n 级。"""
    pal = _wt_palette().astype(np.float64) / 255.0
    x = np.linspace(0.0, 1.0, WT_PALETTE_W)
    xi = np.linspace(0.0, 1.0, n)
    lut = np.empty((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(xi, x, pal[:, c]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return lut


# -------------------------------------------------------------------------
# 线性显示域
# -------------------------------------------------------------------------
# 正版的显示末级是：   level = min(191, raw_uint16 × G >> 10)
# 完全没有对数运算，而且 raw 是**绝对量纲**的（真机实测峰值约 23125 / 25005），
# 不像本工具这样先把峰值归一化到 0 dB。这个差别正是它观感的来源：
#
#   * 没有 log  ⇒  -60 dB（幅度的千分之一）就落到色号 0 = 纯黑，暗场极干净；
#   * 固定增益  ⇒  峰值远超显示上限，min() 把**一整片**裁成 191，
#                  画面上就是"每一刻都有一大片饱和的大红色"。
#
# 本工具把这件事拆成两个正交的参数：
#   1. 感度 —— 线性域的整体增益，直接对应正版的 G；
#   2. 饱和起点 —— 感度的另一种说法（dB 刻度），是显示链路的天然参数。
# 二者是同一个自由度。**对外用「感度」**（正版旋钮的语义就是增益，调大更亮），
# 内部一律用 sat_db（曝光 = 10^(-sat_db/20)），换算只发生在界面层。
#
# 详细换算与锚点说明见下面「感度 ↔ 饱和起点」那一节。
# =========================================================================

# 正版量级的三个参考量。正版末级是 G = 感度 >> 10，raw 是绝对量纲 uint16，
# 所以"把 25000 摆到 191"所需的增益就是下面这个曝光值。


WT_RAW_PEAK = 25000.0       # 真机 uint16 分析输出的典型峰值


WT_SENSE = 100.0            # 正版「感度」默认拉满


WT_LINEAR_EXPOSURE = WT_RAW_PEAK * WT_SENSE / 1024.0 / 191.0   # ≈ 12.7822



# -------------------------------------------------------------------------
# 感度（正版刻度） ↔ 饱和起点（dB，内部刻度）
# -------------------------------------------------------------------------
# 两者是**同一个自由度的两种说法**，形状取自正版：
#
#     G = 感度 × 2^感度档位          （档位就是一个 2 的幂倍率）
#     level = min(191, raw × G >> 10)
#
# 由于本工具用的是**相对峰值**刻度（raw 恒 <= 1），末级的"固定增益 + 夹上限"
# 与"曝光 + 夹到调色板上限"是同一件事，所以：
#
#     感度 = 100 × 10^((sat_db − sat_db@100) / 20)
#
# 为什么对外用「感度」这个名字：正版那个旋钮的语义是**增益**，用户理解的是
# "调大更亮"。而"饱和起点"是个反着说的阈值，调大反而更暗，容易反直觉。
# 内部仍用 sat_db：它是这条链路的天然参数（曝光 = 10^(-sat_db/20)），
# 缓存指纹与配置文件也都按它来。换算只发生在界面层。
#
# ⚠️ 关于锚点：`sat_db@100 = DEFAULT_WT_SAT_DB`，也就是**当前默认观感 = 感度 100**。
# 这是刻意的选择，不是从正版量出来的。原因：真正的绝对标定需要正版 DSP 输出的
# uint16 实际量纲，而本工具的幅度是归一化过的，"sat_db = 0"（曝光 1，不做任何
# 夹断）并不对应正版的 G = 100 + 感度档位。用当前默认当锚点，可以让
#   ① 滑块的默认位置与正版一致（都是 100）；
#   ② 现有观感与所有默认值一个字节都不变。
# 所以这里的 100 应当理解为"本工具的基准感度"，而不是"正版的 G=100"。
# 将来若拿到真实 .wfd 的 uint16 量纲，只需换掉这个锚点即可完成绝对标定。
#
# 另一点如实说明：正版 G 的完整形式还有第二项
# `+ 对比度 × (100 − 背景图透明度)`，本实现**刻意不做**。那一项把显示参数混进
# 亮度增益，会破坏"感度"与"对比度"的正交性；按本工具默认对比度 25 代入，
# G 会涨到 2600（26 倍），整幅直接过曝。


WT_SENSE_AT_100 = DEFAULT_WT_SAT_DB = None      # 占位，下面赋值（避免循环引用）


def wt_exposure_to_sat_db(exposure):
    """线性增益 → 饱和起点(dB)。增益 <= 1 视为不额外加增益。"""
    exposure = float(exposure)
    if not math.isfinite(exposure) or exposure <= 1.0:
        return 0.0
    return float(-20.0 * math.log10(exposure))


def wt_sat_db_to_exposure(sat_db):
    """饱和起点(dB) → 线性增益。仅供换算/排查时查证用，显示链路不调它。"""
    sat_db = float(sat_db)
    if sat_db >= 0.0:
        return 1.0              # 起点抬到 0 dB，等于不额外加增益
    return float(10.0 ** (-sat_db / 20.0))


# 默认饱和起点直接由正版曝光推导，避免手抄魔数造成的偏差。
# 注意它现在的角色：**只作为「感度 100」这个换算锚点**，不再是线性域的默认观感。
DEFAULT_WT_SAT_DB = wt_exposure_to_sat_db(WT_LINEAR_EXPOSURE)   # ≈ -22.1321

# 线性域的**实际默认感度**。
#
# 为什么不直接用上面的锚点（感度 100）：感度 100 对应曝光约 12.8，只有峰值
# 上方约 22 dB 以内的内容不至于全白，实测"过曝、几乎看不出层次"。换到 40.5
# 后曝光约 5.2、可见范围约 14 dB，暗部层次才出得来。
# 这个值必须与界面滑块的默认位置（35%）指向同一个点，两边改动要同步。
DEFAULT_LINEAR_SENSE = 40.5


def wt_default_linear_sat_db():
    """线性域的默认 sat_db（= 感度 DEFAULT_LINEAR_SENSE 对应的 dB）。"""
    return wt_sense_to_sat_db(DEFAULT_LINEAR_SENSE)


# 基准感度 100 就锚在它上面（理由见上面那段说明）。
WT_SENSE_AT_100 = DEFAULT_WT_SAT_DB


def wt_sense_to_sat_db(sense, anchor_db=None):
    """感度（正版刻度，100 = 本工具基准）→ 饱和起点(dB)。

    ⚠️ 方向：感度**越大**，增益越大，饱和起点必须**越小**（越负）。
    因为显示链路里的曝光是 10^(-sat_db/20) —— sat_db 越小曝光越大。
    写成 sat_db = 锚点 − 20·log10(感度/100) 才对；写成加号会让旋钮反向
    （调大反而更暗），这个坑踩过一次。
    """
    a = WT_SENSE_AT_100 if anchor_db is None else float(anchor_db)
    s = float(sense)
    if not (s > 0.0):
        return a
    return float(a - 20.0 * math.log10(s / 100.0))


def wt_sat_db_to_sense(sat_db, anchor_db=None):
    """饱和起点(dB) → 感度（正版刻度，100 = 本工具基准）。"""
    a = WT_SENSE_AT_100 if anchor_db is None else float(anchor_db)
    return float(100.0 * (10.0 ** ((a - float(sat_db)) / 20.0)))


# 滑块量程以基准感度 100 为中心、向两侧各 20 dB —— 覆盖从"只剩骨架"到
# "糊成一片"的全过程，而且两端都是整齐的 10 的幂，滑块刻度不会出现半截。
#
# 注意端点必须满足 WT_SENSE_MIN × WT_SENSE_MAX = 100²（也就是两侧 span 相等），
# 否则滑块的一头会先撞到 WT_SENSE 的上/下限，对应的那一段 dB 永远调不到。
WT_SENSE_MIN = 10.0         # 最暗：sat_db = 基准 − 20
WT_SENSE_MAX = 1000.0       # 最亮：sat_db = 基准 + 20


# dB 端点由感度端点换算，仅供需要 dB 语义的地方查证/夹断用。
# 注意次序：感度最大对应 sat_db **最小**（见上面方向说明），所以这里是反的。
WT_SAT_DB_MIN = wt_sense_to_sat_db(WT_SENSE_MAX)    # 感度 1000 = 基准 − 20 ≈ -42.13
WT_SAT_DB_MAX = wt_sense_to_sat_db(WT_SENSE_MIN)    # 感度 10   = 基准 + 20 ≈ -2.13





LUTS["wavetone"] = _wt_lut()


LUTS_RGB["wavetone"] = (
    LUTS["wavetone"][:, 0].copy(),
    LUTS["wavetone"][:, 1].copy(),
    LUTS["wavetone"][:, 2].copy(),
)
# 主题下拉框的顺序（也是合法配色名的全集）


COLORMAP_CHOICES = ("magma", "inferno", "viridis", "ice", "wavetone")


# =========================================================================
# MIDI / 频率 / 坐标
# =========================================================================
