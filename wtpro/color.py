"""配色：内置配色表、WaveTone 渐变的解码与重采样、线性域饱和起点的换算。"""

import base64

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
#   1. 饱和起点 —— "从多暗开始全部顶格"，直接对应正版的 min() 门限；
#   2. 线性增益 —— 饱和起点以下怎么往上提，由饱和起点反推。
# 二者是同一个自由度的两种说法。这里选「饱和起点」当用户参数，因为 dB 比裸
# 增益好理解、好复现，与正版的对应关系也一目了然：
#
#       gain = 10 ** (-sat_db / 20)          sat_db = -20 * log10(gain)
#
# 滑杆量程取 -60..0 dB：往左到 -60 时 70 dB 以内全部顶格（几乎是"全红"），
# 往右到 0 时只有峰值那一点顶格，中间覆盖了从"糊成一片"到"只剩骨架"的全过程。
# =========================================================================


WT_SAT_DB_MIN = -60.0       # 滑杆下限：饱和起点最深（几乎整幅都顶格）


WT_SAT_DB_MAX = 0.0         # 滑杆上限：只有峰值那一点顶格


SAT_SLIDER_SCALE = 10       # 滑杆是整数，1 格 = 0.1 dB

# 正版量级的三个参考量。正版末级是 G = 感度 >> 10，raw 是绝对量纲 uint16，
# 所以"把 25000 摆到 191"所需的增益就是下面这个曝光值。


WT_RAW_PEAK = 25000.0       # 真机 uint16 分析输出的典型峰值


WT_SENSE = 100.0            # 正版「感度」默认拉满


WT_LINEAR_EXPOSURE = WT_RAW_PEAK * WT_SENSE / 1024.0 / 191.0   # ≈ 12.7822




def wt_exposure_to_sat_db(exposure):
    """线性增益 → 饱和起点(dB)，即 wt_sat_db_to_exposure 的逆。"""
    exposure = float(exposure)
    if not np.isfinite(exposure) or exposure <= 1.0:
        return 0.0              # 增益 <= 1 就没有可饱和的区域了
    return float(-20.0 * np.log10(exposure))




def wt_sat_db_to_exposure(sat_db):
    """饱和起点(dB) → 线性增益。仅供换算/排查时查证用，显示链路不调它。"""
    sat_db = float(sat_db)
    if sat_db >= 0.0:
        return 1.0              # 起点抬到 0 dB，等于不额外加增益
    return float(10.0 ** (-sat_db / 20.0))


# 默认饱和起点直接由正版曝光推导，避免手抄魔数造成的偏差。


DEFAULT_WT_SAT_DB = wt_exposure_to_sat_db(WT_LINEAR_EXPOSURE)   # ≈ -22.1321




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
