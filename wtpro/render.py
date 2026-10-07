"""显示与后处理：dB 换算、显示域映射、阶梯滤镜、对比曲线、色号与 QImage。"""

import math

import numpy as np

from wtpro.color import DEFAULT_WT_SAT_DB, WT_SAT_DB_MAX, WT_SAT_DB_MIN
from wtpro.common import (
    DB_FLOOR, DEFAULT_ROWS_PER_SEMITONE, DISPLAY_GAMMA, F32, F64, MIDI_MAX, _int_opt,
    _num_opt
)
from wtpro.qt import QImage

CURVE_MODES = (
    "none",      # 不压
    "knee",      # 低于阈值的部分乘以增益
    "power",     # 以 pivot 为支点的幂曲线
    "sigmoid",   # 以 center 为中点的 S 曲线
    "wavetone",  # 复刻 WaveTone 的对比度压缩（不含倍音除去）
)

# 显示/曲线的工作域


DOMAIN_MODES = ("db", "linear")

# WaveTone 对比度的取值范围。原版滑块就是 0..100，其中 100 在原版会额外叠加
# 「倍音除去」这一独立功能；本工具不实现该功能，但 0..100 的对比度压缩本身
# 是完整可算的，所以取值上限保留 100。


WT_CONTRAST_MAX = 100



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
    "domain": "db",       # db | linear —— linear 复刻 WaveTone 的线性显示
    "sat_db": DEFAULT_WT_SAT_DB,  # 线性域的饱和起点（dB）
    "wt_contrast": 25,    # WaveTone 对比度 0..WT_CONTRAST_MAX
    # 声道：当前显示哪一张平面（stereo | l | r）。它只是显示状态，
    # 由主窗口在切换时改写；放在默认表里是为了"配置里总有这个键"。
    "channel_plane": "stereo",
}




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
                "sig_center", "sig_k", "sat_db"):
        out[key] = _num_opt(out.get(key), DEFAULT_FILTER_CONFIG[key])
    out["trim"] = _int_opt(out.get("trim"), DEFAULT_FILTER_CONFIG["trim"])
    if out.get("reduce") not in ("mean", "midmax"):
        out["reduce"] = DEFAULT_FILTER_CONFIG["reduce"]
    if out.get("curve") not in CURVE_MODES:
        out["curve"] = DEFAULT_FILTER_CONFIG["curve"]
    if out.get("domain") not in DOMAIN_MODES:
        out["domain"] = DEFAULT_FILTER_CONFIG["domain"]
    # 饱和起点夹到滑杆量程内；NaN/无穷也一并退回默认值
    sat = out["sat_db"]
    if not math.isfinite(sat):
        sat = DEFAULT_WT_SAT_DB
    out["sat_db"] = max(WT_SAT_DB_MIN, min(WT_SAT_DB_MAX, sat))
    out["wt_contrast"] = int(round(_num_opt(out.get("wt_contrast"),
                                            DEFAULT_FILTER_CONFIG["wt_contrast"])))
    out["wt_contrast"] = max(0, min(WT_CONTRAST_MAX, out["wt_contrast"]))
    return out




def _wavetone_contrast(amp, contrast):
    """复刻 WaveTone 2.74 的对比度压缩曲线（**不含倍音除去**）。

    正版的原式（整数运算）：
        l  = contrast*256/400 ; r = 256-l
        v  = (A*r + B*l) >> 8            # A=原始, B=锐化副本；这里没有 B，取 v = A
        ref= max(1, max(v 在该帧的最大值)*75/100)      # ★ 每帧独立
        X  = contrast*256/49 ; w1 = 256-X
        e  = ((v*v/ref)*X + v*w1) >> 8
        level = min(191, (e * ((contrast+100)*G/100)) >> 10)

    **返回值与输入同量纲**（都是"线性幅度"），不做任何峰值归一化 —— 这一步很关键：
    正版的末级是 `level = min(191, e*G2 >> 10)`，是**固定增益 + 饱和截断**。
    因为 min() 的存在，e 高于某个门限的**一整片**都会变成 191，画面上就是
    "每一刻都有一片大红"。如果在这里把结果按峰值归一化成 1.0，就只有最强的
    **单个点**到顶，红色几乎消失 —— 那是错的。

    所以这里只做两件事：
      1. 按正版的二次曲线整形（ref 逐帧）；
      2. 乘上末级增益比 (contrast+100)/100（对应原式的 G2 = (contrast+100)*G/100）。
    整体曝光交给 db_to_u8 的饱和起点统一处理。

    关键性质：
      * ref 是**每一帧自己峰值的 75%**（线性域），门槛跟着最响的音走 —— 这就是
        "原本明显的音亮度不变、旁边暗的逐渐消失"的原因；
      * 对同一个 contrast，输出随输入**单调不减**；
      * 末级增益比 (contrast+100)/100 随 contrast 增大，所以峰值会被略微抬高
        （contrast=100 时约 +10.5 dB）。原式同样是先整形再乘增益，属于原版行为；
      * contrast > 49 时 X > 256 ⇒ w1 < 0，线性项变负号：从"二次混合"变成
        "先减掉一个底噪再平方"，效果明显加速，并出现真正的零点截断；
      * contrast == 0 时原样返回（e 退化成 v，增益比为 1）；
      * contrast == 100 时抛物线在 v = ref·8/13（约 0.4615·peak）处过零，低于该值的
        部分全部输出 0 并压至显示下限；峰值附近保留约 8.4 dB 的层次。

    与正版的差别只在 contrast == 100 时：原版会在此额外叠加「倍音除去」这一独立
    功能，本工具不实现。因此 100 在本工具里表示"对比度压缩本身的极限"，
    而不再是"切换到另一个功能"。

    amp: 线性幅度（全局峰值归一 = 1.0），形状 (n_frames, n_freq)。
    """
    con = max(0, min(WT_CONTRAST_MAX, int(contrast)))
    v = np.asarray(amp, dtype=np.float64)
    if con <= 0:
        return v
    X = (con * 256) // 49
    w1 = 256 - X
    peak = np.maximum(v.max(axis=1, keepdims=True), 1e-12)
    ref = peak * 0.75                           # 逐帧峰值 × 75%
    e = ((v * v / ref) * X + v * w1) / 256.0
    # con 较大时线性项转负，e 会取到 0 甚至负值。夹到 0：调用方再夹一次下限，
    # 相当于把这些点压到显示下限。不夹的话负值会进 log10 变成 NaN。
    # 返回值量纲与输入一致；由于末级增益比可到 2.0，峰值会被抬到约 +6 dB，
    # 叠加二次项在峰值极低的帧上的相对提升，实测总抬升不超过约 +11 dB。
    # 这是原式"先整形再乘增益"的固有行为，下游 db_to_u8 会把 >0 dB 的部分 clip 掉。
    return np.maximum(e, 0.0) * ((con + 100) / 100.0)   # 末级增益比 G2/G




def _apply_curve(mag_db, c):
    """对归约后的 dB 施加所选对比曲线，返回同形状的 dB。

    统一约束：曲线**只压不强推**，且必须**单调不减** —— 输入更亮的结果不能
    更暗，输入更暗的结果不能更亮。否则画面上会出现反直觉的明暗倒挂。
    除 wavetone 外都满足 out <= in 且 out(0 dB) == 0 dB（峰值不受影响）。

    knee    : 线性幅度低于阈值的部分乘以增益 g。
    power   : 以 pivot 为支点，pivot 以下按 gamma 压缩，以上保持原值。
    sigmoid : dB 域 S 形软限幅。以 center 为拐点，比它暗的部分被压向显示下限、
              比它亮的部分保持原值，中间是平滑过渡（不会像 knee 那样在阈值处
              留一条硬折线）。
    wavetone: 复刻正版的对比度压缩。在**线性幅度**域做二次整形，逐帧以自己的
              峰值为参考，所以峰值不动、弱音被逐步压没。注意它为了复刻原式，
              末尾会乘一个 >1 的增益比，因此不保证 out <= in。
    """
    mode = c["curve"]
    if mode == "wavetone":
        con = int(round(_num_opt(c.get("wt_contrast"),
                                 DEFAULT_FILTER_CONFIG["wt_contrast"])))
        if con <= 0:
            return mag_db
        amp = np.power(10.0, np.asarray(mag_db, dtype=np.float64) / 20.0)
        shaped = _wavetone_contrast(amp, con)
        # 下限取 1e-12（-240 dB）：对比度拉满时会有成片的点被压到 0，
        # 这里统一落到远低于 DB_FLOOR 的值，再由 maximum 收敛到显示下限。
        return np.maximum(20.0 * np.log10(np.maximum(shaped, 1e-12)),
                          float(DB_FLOOR))
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




TRIM_PRESETS = {1: 0, 2: 1, 3: 1, 4: 1, 6: 1, 8: 2, 12: 3, 16: 4, 24: 6, 32: 8, 48: 12}




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




def db_to_u8(db, floor_db=DB_FLOOR, hide_low=0.0, gamma=DISPLAY_GAMMA,
             domain="db", sat_db=DEFAULT_WT_SAT_DB):
    """dB → 0..255 的色号。

    两个域的区别只在第一层映射：

    domain="db"（默认）
        在 dB 域线性铺开：[-100, 0] dB 均匀对应整个调色板。
        动态范围完整呈现，低声压级内容同样可见，观感接近 Audacity 一类工具。

    domain="linear"
        **复刻 WaveTone**：直接按线性幅度映射，全程不作对数运算。
            norm = 10^(db/20) × gain,  gain = 10^(-sat_db/20)
        sat_db 为饱和起点：高于该电平的部分一律取调色板上限（饱和），
        低于该电平的部分按指数关系迅速衰减至调色板下端。默认 -22.13 dB
        对应 WaveTone 的固定增益 12.78，故默认取值即正版观感（界面上显示为
        -22.1，滑杆精度 0.1 dB）。映射为纯指数关系，-60 dB（幅度的千分之一）
        即落到色号 0，因此暗场压缩明显、峰值区域成片饱和。

    sat_db 支持标量或与 db 同形状的数组（后者供调试与实验用）。
    其后的 hide_low（亮度滤镜）与 gamma 由两个域共用。
    """
    db = np.asarray(db, dtype=F32)
    if str(domain) == "linear":
        # 饱和起点反推增益。三种退化情况都让这一档"透明"（增益 1、起点 0 dB）：
        # sat_db 未设、非有限、或 >= 0（起点抬到 0 dB 就没有可饱和的区域了）。
        # 先夹到 -400 dB 只是为了让 10**(-sat/20) 不溢出（那个值本来也会被 keep
        # 挡掉），夹完的结果始终有限，所以不必再操心浮点告警。
        sat = np.asarray(sat_db, dtype=F32)
        if sat.ndim > 0 and sat.shape != db.shape:
            sat = np.full(db.shape, float(np.ravel(sat)[0]), dtype=F32)
        keep = np.isfinite(sat) & (sat < 0.0)
        safe = np.where(keep, np.maximum(sat, F32(-400.0)), F32(1.0))
        expo = np.power(F32(10.0), -safe * F32(1.0 / 20.0), dtype=F32)
        norm = np.power(F32(10.0), db * F32(1.0 / 20.0), dtype=F32)
        norm *= expo
        np.clip(norm, F32(0.0), F32(1.0), out=norm)
    else:
        span = max(1e-9, -float(floor_db))
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
#               五种曲线都只压不强推，不做任何提亮。
# 只作用于显示，不改动分析结果和导出的数据。
# =========================================================================


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
