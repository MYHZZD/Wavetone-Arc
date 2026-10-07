"""各类设置对话框：分析参数、阶梯滤镜、谱面设置、和弦编辑。"""

import math

import numpy as np

from wtpro import backends
from wtpro.color import (
    COLORMAP_CHOICES, DEFAULT_WT_SAT_DB, SAT_SLIDER_SCALE, WT_SAT_DB_MAX, WT_SAT_DB_MIN
)
from wtpro.common import (
    BLACK_PC, CUSTOM_CHORD_HI, CUSTOM_CHORD_LO, DEFAULT_CHANNEL_MODE,
    DEFAULT_ROWS_PER_SEMITONE, DEFAULT_TARGET_FPS, DEFAULT_WINDOW, FPS_CHOICES,
    KEY_SPAN_HI_MIN, KEY_SPAN_LO_MAX, MIDI_MAX, MIDI_MIN, MIN_KEY_SPAN,
    MODE_PLANES, PLANE_LABELS, ROWS_PER_SEMITONE_CHOICES, WINDOW_CHOICES,
    _num_opt, channel_mode_from_flags, midi_name
)
from wtpro.render import WT_CONTRAST_MAX, _filter_config, scaled_trim
from wtpro.qt import (
    QCheckBox, QColor, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFont,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPainter, QPen, QPointF, QPushButton,
    QRectF, QSizePolicy, QSlider, QSpinBox, QTimer, QVBoxLayout, QWidget, Qt,
    pyqtSignal
)

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

    def isEnabled(self):
        return self.slider.isEnabled()

    def refresh_label(self):
        self.value_label.setText(self.fmt.format(self.value()))




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
        self.cb_window.setToolTip(
            "频谱分析的窗函数，用于抑制截断引起的频谱泄漏，各项取舍如下。\n"
            "  hann              主瓣最窄，通用性最好，适用于大多数素材。\n"
            "  blackman-harris   旁瓣抑制优于 hann，弱音与强音的分离度更好。\n"
            "  blackman-harris-7 旁瓣抑制进一步增强，代价是主瓣变宽。\n"
            "  kaiser            主瓣宽度与旁瓣幅度的折中选择。\n"
            "  hamming           旁瓣衰减较慢，一般不作首选。\n"
            "  flattop           幅度标定最准确，但频率分辨率最低。")
        layout.addRow("窗函数", self.cb_window)

        self.cb_rows = QComboBox()
        for r in ROWS_PER_SEMITONE_CHOICES:
            self.cb_rows.addItem(str(r), r)
        cur_r = params.get("rows_per_semitone", DEFAULT_ROWS_PER_SEMITONE)
        try:
            self.cb_rows.setCurrentIndex(ROWS_PER_SEMITONE_CHOICES.index(cur_r))
        except ValueError:
            self.cb_rows.setCurrentIndex(1)
        self.cb_rows.setToolTip(
            "频谱图纵轴的分辨率，即每个半音占用多少行。\n"
            "数值增大：音高方向的层次更细，半音内部的变化更平滑，\n"
            "            但总行数与显存占用按比例上升。\n"
            "数值减小：一个半音内的各行趋于一致，半音之间的边界更分明，\n"
            "            总行数减少，绘制更快。\n"
            "阶梯滤镜的归约与去边行数 N 均以此为基准。")
        layout.addRow("每半音行数", self.cb_rows)

        self.cb_fps = QComboBox()
        for f in FPS_CHOICES:
            self.cb_fps.addItem(str(f), f)
        cur_f = params.get("target_fps", DEFAULT_TARGET_FPS)
        try:
            self.cb_fps.setCurrentIndex(FPS_CHOICES.index(cur_f))
        except ValueError:
            self.cb_fps.setCurrentIndex(2)
        self.cb_fps.setToolTip(
            "频谱图横轴的时间分辨率，即每秒生成多少帧。\n"
            "数值增大：起音瞬间与颤音的变化更清晰，分析耗时与内存占用\n"
            "            按比例上升，总帧数上限为 65536。\n"
            "数值减小：出图更快、文件更小，短促的瞬态可能被跨越而丢失。")
        layout.addRow("时间分辨率 (fps)", self.cb_fps)

        # ---- 声道分析 ----
        # 这里的勾选决定"算哪些声道"，与「谱面设置 - 外观 - 声道」下拉框
        # （决定"看哪一张"）是两件事。
        grp_ch = QGroupBox("声道分析")
        fc = QVBoxLayout(grp_ch)
        row_ch = QHBoxLayout()
        row_ch.setContentsMargins(0, 0, 0, 0)

        self.chk_ch_l = QCheckBox("左声道 (L)")
        self.chk_ch_l.setChecked(bool(params.get("channel_l", False)))
        self.chk_ch_r = QCheckBox("右声道 (R)")
        self.chk_ch_r.setChecked(bool(params.get("channel_r", False)))
        self.chk_ch_l.setToolTip(
            "单独分析左声道。单声道输入（1 路）时与右声道等价 —— 两路取到的是同一份数据。")
        self.chk_ch_r.setToolTip(
            "单独分析右声道。单声道输入（1 路）时与左声道等价 —— 两路取到的是同一份数据。")
        row_ch.addWidget(self.chk_ch_l)
        row_ch.addWidget(self.chk_ch_r)
        row_ch.addStretch(1)
        fc.addLayout(row_ch)

        self.lbl_ch_hint = QLabel("")
        self.lbl_ch_hint.setWordWrap(True)
        self.lbl_ch_hint.setStyleSheet("color:#9fc5ff;")
        fc.addWidget(self.lbl_ch_hint)

        self._ch_hint_tip = (
            "都不勾：只算一次，两路取算术平均（即原有行为），出 1 张图。\n"
            "  这一路是**代数**平均，先相加再分析：反相内容会互相抵消而消失。\n"
            "只勾一路：只算一次，只分析该声道，出 1 张图。\n"
            "两路都勾：算两次（L 一次、R 一次），并由 |L| 与 |R| 的**幅度**平均\n"
            "  得出 stereo，共 3 张图。这一路不会抵消，所以反相内容仍然可见。\n"
            "勾选后实际会算出哪些平面：\n"
            "  都不勾  → stereo\n"
            "  只勾 L  → L\n"
            "  只勾 R  → R\n"
            "  都勾    → stereo / L / R\n"
            "\n"
            "最多只跑两次分析 —— stereo 由 L、R 后处理平均得出，不会为了它再跑一遍。\n"
            "看哪一张由「谱面设置 - 外观 - 声道」下拉框决定。")
        self._sync_channel_hint()
        self.chk_ch_l.toggled.connect(self._sync_channel_hint)
        self.chk_ch_r.toggled.connect(self._sync_channel_hint)
        layout.addRow(grp_ch)

        # ---- 插值 ----
        # 重分配的输出落在离散的 (帧, bin) 网格上，要搬到显示的 (帧, 音高行)
        # 网格上。两个方向的间距并不整除，必须插值才能连续。
        grp_norm = QGroupBox("插值")
        fn = QFormLayout(grp_norm)

        self.chk_interp = QCheckBox("启用插值（频率方向 + 时间方向）")
        self.chk_interp.setChecked(bool(params.get("interp", True)))
        self.chk_interp.setToolTip(
            "重分配的插值开关，同时作用于频率与时间两个方向。\n"
            "频率方向（bin → 音高行）：每个 bin 的幅度按高斯核撒到相邻的行上。\n"
            "时间方向（各频段网格 → 公共帧率）：按高斯核重采样到统一的帧率。\n"
            "核宽自适应：σ = 2.0 × bin 宽，再按该行的行宽换算成行数。\n"
            "  这样相邻 bin 的高斯覆盖始终有足够重叠，行网格不会落进缝隙；\n"
            "  σ 恒等于物理分辨率（bin 宽），因此不会过度模糊。\n"
            "启用（推荐）：相邻行亮度差的中位数约 1.3 dB，画面连续。\n"
            "禁用：每个点直接落到最近的行/帧，画面最锐利，但会出现横向的\n"
            "      梳状明暗条纹（相邻行差中位数约 10 dB，超过 6 dB 的占 68%）。\n"
            "修改后需重新分析。")
        fn.addRow(self.chk_interp)

        layout.addRow(grp_norm)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _channel_mode(self):
        """当前勾选对应的分析模式。"""
        return channel_mode_from_flags(self.chk_ch_l.isChecked(),
                                       self.chk_ch_r.isChecked())

    def _sync_channel_hint(self, *_a):
        """刷新提示文字与 tooltip：勾选组合一变，代价和产出平面都跟着变。"""
        mode = self._channel_mode()
        planes = MODE_PLANES.get(mode, ("stereo",))
        runs = 2 if mode == "lr_both" else 1
        names = "/".join(PLANE_LABELS.get(p, p) for p in planes)
        self.lbl_ch_hint.setText(
            f"分析{runs}次 频谱图{len(planes)}张\n{names}")
        self.lbl_ch_hint.setToolTip(self._ch_hint_tip)

    def get_params(self):
        want_l = bool(self.chk_ch_l.isChecked())
        want_r = bool(self.chk_ch_r.isChecked())
        return {
            "window_name": self.cb_window.currentText(),
            "rows_per_semitone": int(self.cb_rows.currentData()),
            "target_fps": int(self.cb_fps.currentData()),
            "interp": bool(self.chk_interp.isChecked()),
            # 分析侧只认 channel_mode；两个布尔量留着是为了让"用户勾了什么"
            # 在配置里可读，worker 拿不到 mode 时也能自己推出来。
            "channel_mode": channel_mode_from_flags(want_l, want_r),
            "channel_l": want_l,
            "channel_r": want_r,
        }


# =========================================================================
# 只在松手时汇报的滑块
# =========================================================================


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
        ("不压缩", "none"),
        ("knee：低于阈值整体下移", "knee"),
        ("幂曲线：以支点连续压缩", "power"),
        ("S 形：拐点以下压向底", "sigmoid"),
        ("WaveTone 对比度：峰值不变、弱音压至底", "wavetone"),
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
            "归约方式：把每个半音内的多行压缩为一个特征值，作为该半音的台阶高度。\n"
            "取平均：以行均值作为特征值，各半音能量呈平滑过渡，\n"
            "        但半音内的峰值会被摊薄，亮点减弱。\n"
            "去边取最大：去掉最上、最下各 N 行后取最大值，保留峰值的同时\n"
            "        排除边界处的频谱泄漏，台阶边界更明确。\n"
            f"当前每个半音 {max(1, int(rows_per_semitone))} 行"
            f"{'，仅 1 行时去边取最大无意义，该项已禁用' if int(rows_per_semitone) <= 1 else ''}。")
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
            "去边取最大时，从每个半音的最上、最下各去掉的行数 N。\n"
            "N = 0：等同于整段取最大，不做任何剔除。\n"
            "N 增大：界限向半音中心收拢，边界泄漏被剔除得更多，台阶更分明；\n"
            "        取值过大则将没有行可供取样。\n"
            f"当前每个半音 {self._rps} 行，N 的最大值为 {self._trim_max}。\n"
            "峰值通常位于半音中部，泄漏集中在上下边界，因此剔除边界的收益最高。\n"
            "预设值按 12 行去 3 行的比例折算：6 行取 1，24 行取 6，48 行取 12。")
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
            "对比曲线：对归约结果再作一次映射，把较弱的半音继续压低，\n"
            "半音之间的台阶更陡。\n"
            "各项的调整逻辑：\n"
            "  不压：保持归约结果不变。\n"
            "  knee：低于阈值的部分整体乘以一个固定增益。\n"
            "  幂曲线：以支点为界，对支点以下作幂压缩。\n"
            "  S 形：以拐点为中点、按 dB 作平滑软限幅。\n"
            "  WaveTone 对比度：逐帧以本帧峰值为参考作二次整形。\n"
            "全部曲线只压缩、不提升，以避免过曝。")
        self.cb_curve.currentIndexChanged.connect(self._apply_now)
        root.addRow("曲线", self.cb_curve)

        # ---- 各曲线的滑块 ----
        # on_change 只负责刷新数字；真正重算挂在松手上（ReleaseSlider）。
        # 说明里的数值都是线性幅度（0..1），实现按 dB 折算：
        # 幅度 a 对应 20·lg(a) dB。
        self.s_knee_th = _SliderRow(
            root, "knee 阈值", 0.0, 1.0, c["knee_th"], "{:.3f}",
            "knee 曲线的阈值，以线性幅度（0..1）给定：低于该值的部分整体乘以\n"
            "knee 增益，高于该值的部分保持不变。\n"
            "数值增大：参与压缩的范围随之扩大，画面整体转暗。\n"
            "单位换算：幅度 a 对应 20·lg(a) dB，如 0.15 对应 -16.5 dB。",
            on_change=self._enabled_only, on_release=self._changed)
        self.s_knee_g = _SliderRow(
            root, "knee 增益", 0.0, 1.0, c["knee_gain"], "{:.2f}",
            "knee 增益：施加于 knee 阈值以下的固定增益。\n"
            "1.0：不压缩；0.5：整体下降约 6 dB；0：压至显示下限。\n"
            "该增益对整段取值相同，因此不会随信号变弱而加深压缩。",
            on_change=self._enabled_only, on_release=self._changed)

        self.s_pow_g = _SliderRow(
            root, "幂 gamma", 1.0, 6.0, c["pow_gamma"], "{:.2f}",
            "幂 gamma：幂曲线的压缩强度指数。\n"
            "1.0：不压缩；数值增大：支点以下的暗部被推向更低的电平。\n"
            "与 knee 的区别：knee 在阈值处形成折点，幂曲线为连续压缩。",
            on_change=self._enabled_only, on_release=self._changed)
        self.s_pow_p = _SliderRow(
            root, "幂支点", 0.01, 1.0, c["pow_pivot"], "{:.3f}",
            "幂曲线的支点，以线性幅度（0..1）给定：等于该值的电平不变，\n"
            "低于该值的部分按 gamma 压缩，高于该值的部分保持原值。\n"
            "支点升高：压缩范围向亮部扩展；支点降低：压缩仅作用于更暗的部分。\n"
            "单位换算：幅度 a 对应 20·lg(a) dB，如 0.3 对应 -10.5 dB。\n"
            "支点以上不作提升，以避免过曝。",
            on_change=self._enabled_only, on_release=self._changed)

        self.s_sig_c = _SliderRow(
            root, "S 拐点", 0.0, 1.0, c["sig_center"], "{:.3f}",
            "软限幅的拐点位置，以显示范围的比例给定：0 对应峰值（0 dB），\n"
            "1 对应显示下限（-100 dB）。\n"
            "拐点及其以下：压缩量最大；拐点以上：压缩量逐渐减至 0。\n"
            "数值增大：参与压缩的电平范围上移，更多内容被压暗。\n"
            "0 表示不压缩，等同于关闭该曲线。",
            on_change=self._enabled_only, on_release=self._changed)
        self.s_sig_k = _SliderRow(
            root, "S 过渡宽度", 0.02, 1.0, c["sig_k"], "{:.2f}",
            "S 过渡宽度：软限幅过渡带的宽度，以拐点深度的比例给定。\n"
            "数值增大：过渡更平缓，压缩效果接近线性压暗。\n"
            "数值减小：过渡更陡峭，压缩效果趋近于硬阈值。\n"
            "与 knee 的区别即在于此：knee 为分段折线，S 形为连续过渡。",
            on_change=self._enabled_only, on_release=self._changed)

        self.s_wt_c = _SliderRow(
            root, "WaveTone 对比度", 0, WT_CONTRAST_MAX, c["wt_contrast"], "{:.0f}",
            "WaveTone 对比度：WaveTone 对比度参数的复刻，取值范围 0..100。\n"
            "调整逻辑：以本帧峰值的 75% 作为参考电平，对线性幅度作二次整形，\n"
            "因此门槛随各帧最强音而移动。\n"
            "0：保持原值。\n"
            "数值增大：低于参考电平的弱音依次被压至显示下限，峰值被略微抬高\n"
            "（末级增益比 (对比度+100)/100，100 时约 +10.5 dB），取值较大时\n"
            "仅保留最强的一条谱线。\n"
            "取值大于 49 时线性项系数转负，压缩速度明显加快。\n"
            "100：约 46% 峰值以下全部压黑，峰值附近保留约 8.4 dB 的层次。\n"
            "原版在 100 时会额外叠加「倍音除去」功能，本工具不实现该功能，\n"
            "因此这里的 100 表示对比度压缩自身的极限。",
            on_change=self._enabled_only, on_release=self._changed)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_reset = QPushButton("恢复默认")
        self.btn_reset.setToolTip(
            "把本窗口的全部选项恢复为初始值：归约方式取最大、去边行数 N 按\n"
            "每半音行数折算、曲线不压缩、WaveTone 对比度 25。\n"
            "各项滑块回到默认值时，频谱图立即按新参数重绘。")
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
            "wt_contrast": int(round(self.s_wt_c.value())),
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
        self.s_wt_c.setEnabled(cur == "wavetone")

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
        self.s_wt_c.set_value(c["wt_contrast"])
        for s in (self.s_knee_th, self.s_knee_g, self.s_pow_g, self.s_pow_p,
                  self.s_sig_c, self.s_sig_k, self.s_wt_c):
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
        self.btn_clear.setToolTip("清空当前选中的全部和弦音。")
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
            "钢琴窗与频谱图显示音域的最低音。\n"
            f"可调范围：{midi_name(MIDI_MIN)} – {midi_name(KEY_SPAN_LO_MAX)}。\n"
            "数值增大：隐藏最低端的半音，其余内容在同等窗口高度下被拉高，\n"
            "          局部细节更易辨识。\n"
            "数值减小：显示更多低音，单个半音占用的高度相应减少。")
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
            "钢琴窗与频谱图显示音域的最高音。\n"
            f"可调范围：{midi_name(KEY_SPAN_HI_MIN)} – {midi_name(MIDI_MAX)}。\n"
            "数值减小：隐藏最高端的半音，其余内容在同等窗口高度下被拉高，\n"
            "          局部细节更易辨识。\n"
            "数值增大：显示更多高音，单个半音占用的高度相应减少。")
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
        self.btn_key_full.setToolTip(
            "把最低音与最高音恢复为全键盘范围，取消当前的音域裁剪。")
        self.btn_key_full.clicked.connect(self._reset_key_range)
        fk.addRow(self.btn_key_full)
        root.addRow(grp_key)

        # ---------------- 外观 ----------------
        grp_look = QGroupBox("外观")
        fl = QFormLayout(grp_look)
        self.cb_cmap = QComboBox()
        for name in COLORMAP_CHOICES:
            self.cb_cmap.addItem(name, name)
        self.cb_cmap.setToolTip(
            "频谱图的颜色映射（由暗到亮依次取用的调色板）。\n"
            "magma / inferno / viridis：感知均匀的通用配色，暗部过渡平滑。\n"
            "ice：以青白色系表现高强度区域。\n"
            "wavetone：取自 WaveTone 2.74 自带的 wtgcolor.bmp，为该版本的默认\n"
            "          渐变（黑 → 青 → 红），共 192 级，与正版一致。")
        fl.addRow("主题", self.cb_cmap)

        # 声道：显示哪一张平面图。候选由「分析参数」里勾了哪些声道决定，
        # 这里只负责选，不改分析结果。顺序固定 stereo → L → R。
        self.cb_channel = QComboBox()
        self.cb_channel.setToolTip(
            "显示哪一张平面图。候选由「分析参数 - 声道分析」的勾选决定：\n"
            "  都不勾（stereo）：只有「立体声（L/R 平均）」一项。\n"
            "  只勾 L / 只勾 R   ：只有对应的一项。\n"
            "  两路都勾          ：立体声 / 左声道 / 右声道 三项。\n"
            "切换只换显示的那一张，不会重新分析；播放源会跟着换成对应的声道。\n"
            "默认优先选立体声，没有立体声平面时选左声道。")
        fl.addRow("声道", self.cb_channel)

        # 显示域：dB 域（原有行为）/ 线性幅度（复刻 WaveTone）
        self.cb_domain = QComboBox()
        self.cb_domain.addItem("dB 域（动态范围完整，暗部可见）", "db")
        self.cb_domain.addItem("线性幅度（WaveTone 风格，暗场近纯黑）", "linear")
        self.cb_domain.setToolTip(
            "显示域：亮度值经何种映射换算为调色板色号。\n"
            "dB 域：以 dB 为单位线性铺开，[-100, 0] dB 均匀对应整个调色板。\n"
            "        动态范围完整呈现，低声压级内容（如背景噪声）同样可见，\n"
            "        观感接近 Audacity 一类工具。\n"
            "线性幅度：直接以线性幅度映射，全程不作对数运算，为 WaveTone 的\n"
            "        原版行为。映射为指数关系，其饱和起点由下方的「饱和起点」\n"
            "        滑块设定；低于该起点的内容迅速衰减至调色板下端。\n"
            "        典型效果为暗场近乎纯黑、峰值区域成片饱和，对比强烈。")
        fl.addRow("显示域", self.cb_domain)

        # 线性域的饱和起点：比它响的部分一律顶格
        self.sld_sat = ReleaseSlider(Qt.Horizontal, self)
        self.sld_sat.setRange(int(round(WT_SAT_DB_MIN * SAT_SLIDER_SCALE)),
                              int(round(WT_SAT_DB_MAX * SAT_SLIDER_SCALE)))
        self.sld_sat.setToolTip(
            "线性幅度域的饱和起点：高于该电平的部分一律取调色板上限（饱和），\n"
            "低于该电平的部分按指数关系迅速衰减至调色板下端。\n"
            "调整方向：\n"
            "  向左（更负）：饱和范围扩大，亮部连成一片，层次减少。\n"
            "  向右（趋近 0 dB）：仅峰值附近饱和，画面转暗，谱线更锐利。\n"
            "默认 -22.1 dB（精确值 -22.1321 dB）对应 WaveTone 的固定增益 12.78，\n"
            "即其末级 min(191, raw×G>>10) 的截断门限，故默认取值即正版观感。\n"
            "该参数仅在显示域为「线性幅度」时生效。")
        row_sat = QHBoxLayout()
        row_sat.setContentsMargins(0, 0, 0, 0)
        row_sat.addWidget(self.sld_sat, 1)
        self.lbl_sat_local = QLabel("")
        self.lbl_sat_local.setMinimumWidth(52)
        self.lbl_sat_local.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_sat_local.setStyleSheet(
            "color:#9fc5ff;font-family:Consolas,Menlo,monospace;")
        row_sat.addWidget(self.lbl_sat_local)
        fl.addRow("饱和起点", row_sat)

        # 亮度滤镜滑块。**必须由对话框自己创建**：
        # 它天生以对话框为 parent，窗口关闭时不会被顺手删掉（deleteLater 只删
        # 对话框本身，子控件会被 parent 机制留下），主窗口可以放心继续持有它。
        # 反过来把工具栏的滑块 addWidget 进来则会被重设 parent 到对话框，
        # 窗口一关对象就没了 —— 那正是之前 RuntimeError 的原因。
        self.sld_hide = ReleaseSlider(Qt.Horizontal, self)
        self.sld_hide.setRange(0, 100)
        self.sld_hide.setToolTip(
            "亮度滤镜：作完显示域映射后，把低于指定比例的电平统一映射到调色板\n"
            "下端（0% 保持原始映射范围不变）。\n"
            "调整后各域的等效门限不同（下文按默认饱和起点 -22.1321 dB 计算）：\n"
            "  dB 域：0% → -100 dB，35% → -65 dB，100% → 0 dB。\n"
            "  线性幅度：0% → -70.3 dB，35% → -31.3 dB，100% → -22.1 dB。\n"
            "可见切换显示域会改变该滑块的等效作用深度。")
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
        self.chk_beats.setToolTip(
            "在频谱图上叠加节拍线与小节线，用于比对音符与节奏位置。\n"
            "启用后可调下方 BPM 与拍/小节；关闭时两项一并禁用。")
        self.chk_beats.toggled.connect(self._sync_enabled)
        fb.addRow(self.chk_beats)

        self.sp_bpm = QDoubleSpinBox()
        self.sp_bpm.setRange(20.0, 400.0)
        self.sp_bpm.setDecimals(2)
        self.sp_bpm.setSingleStep(1.0)
        self.sp_bpm.setToolTip(
            "每分钟拍数，用于确定节拍线的间距。\n"
            "数值增大：节拍线加密；数值减小：节拍线变疏。\n"
            "间距以 BPM 换算为帧序号后绘制，帧跳跃长度由时间分辨率（fps）\n"
            "决定并量化到 2 的幂，因此修改 fps 会同时改变节拍线的实际位置。\n"
            "缩放比例过小时（每拍不足 6 像素）节拍线自动隐藏。")
        fb.addRow("BPM", self.sp_bpm)

        self.sp_bpb = QSpinBox()
        self.sp_bpb.setRange(1, 16)
        self.sp_bpb.setToolTip(
            "每小节的拍数，用于确定小节线的间距。\n"
            "数值增大：小节线变疏；数值减小：小节线变密。\n"
            "仅在启用「显示节拍线与小节线」后生效。\n"
            "缩放比例过小时（每小节不足 6 像素）小节线自动隐藏。")
        fb.addRow("拍/小节", self.sp_bpb)
        root.addRow(grp_beat)

        # ---------------- 按钮 ----------------
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_reset = QPushButton("恢复默认")
        self.btn_reset.setToolTip(
            "把本窗口的全部选项恢复为初始值：全键盘音域、magma 主题、\n"
            "dB 域、亮度滤镜 35%、启用节拍线、BPM 120、每小节 4 拍。")
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
        di = self.cb_domain.findData(s.get("domain", "db"))
        self.cb_domain.setCurrentIndex(max(0, di))
        # 分析模式决定声道下拉框的候选；s 里没有就问主窗口要，再没有就用默认
        self._mode = s.get("channel_mode", DEFAULT_CHANNEL_MODE)
        self._sync_channel_combo(keep=s.get("channel_plane"))
        self.sld_sat.setValueSilently(
            self._sat_db_to_slider(s.get("sat_db", DEFAULT_WT_SAT_DB)))
        self.chk_beats.setChecked(bool(s.get("show_beats", True)))
        self.sp_bpm.setValue(float(s.get("bpm", 120.0)))
        self.sp_bpb.setValue(int(s.get("beats_per_bar", 4)))
        self.sld_hide.setValueSilently(int(s.get("volume_percent", 35)))
        if not has_data:
            grp_look.setEnabled(False)
            grp_beat.setEnabled(False)
        self._loading = False

        self.cb_cmap.currentIndexChanged.connect(self._apply)
        self.cb_domain.currentIndexChanged.connect(self._apply)
        self.cb_channel.currentIndexChanged.connect(self._apply)
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
        self.sld_sat.valueChanged.connect(self._on_sat_label)
        self.sld_sat.valueReleased.connect(self._on_sat_released)

        self._refresh_key_labels()
        self._on_volume_label(self.sld_hide.value())
        self._on_sat_label(self.sld_sat.value())
        self._sync_enabled()
        self.resize(480, self.sizeHint().height())

    # ---------------- 声道 ----------------
    def _channel_mode(self):
        """当前生效的分析模式（来自主窗口保存的分析参数）。"""
        return getattr(self, "_mode", DEFAULT_CHANNEL_MODE)

    def _channel_planes(self):
        """按当前分析模式算出可选平面，顺序固定 stereo → l → r。"""
        return list(MODE_PLANES.get(self._channel_mode(), ("stereo",)))

    def _sync_channel_combo(self, keep=None):
        """按分析模式重建声道下拉框。

        keep 是希望保留的平面名；它已不在候选里（比如刚取消了 L 的勾选）
        就退回候选的第一个，也就是"有 stereo 就用 stereo"。
        """
        planes = self._channel_planes()
        if keep is None:
            keep = self.cb_channel.currentData()
        self._loading = True
        try:
            self.cb_channel.clear()
            for p in planes:
                self.cb_channel.addItem(PLANE_LABELS.get(p, p), p)
            idx = self.cb_channel.findData(keep) if keep else -1
            if idx < 0:
                idx = 0
            self.cb_channel.setCurrentIndex(max(0, idx))
        finally:
            self._loading = False
        self.cb_channel.setEnabled(bool(planes))

    def _current_channel_plane(self):
        data = self.cb_channel.currentData()
        if data:
            return data
        planes = self._channel_planes()
        return planes[0] if planes else "stereo"

    def set_channel_mode(self, mode):
        """主窗口在重新分析后告知新的分析模式，据此刷新候选平面。

        当前选中的平面若仍在新候选里就保留（例如从 lr_both 换成 l_only 时
        正看着 L，就继续看 L）；否则退回第一个候选（有 stereo 就用 stereo）。
        """
        mode = mode or DEFAULT_CHANNEL_MODE
        if mode == getattr(self, "_mode", None):
            return
        self._mode = mode
        self._sync_channel_combo(keep=self.cb_channel.currentData())
        self._apply()

    # ---------------- 饱和起点 ----------------
    def _sat_db_to_slider(self, sat_db):
        """饱和起点(dB) → 滑块整数，并夹到量程内。"""
        sat = _num_opt(sat_db, DEFAULT_WT_SAT_DB)
        if not math.isfinite(sat):
            sat = DEFAULT_WT_SAT_DB
        sat = max(WT_SAT_DB_MIN, min(WT_SAT_DB_MAX, sat))
        return int(round(sat * SAT_SLIDER_SCALE))

    def _sat_db(self):
        return self.sld_sat.value() / float(SAT_SLIDER_SCALE)

    def _on_sat_label(self, _v=None):
        """拖动中只更新数字，不做任何重算。"""
        self.lbl_sat_local.setText(f"{self._sat_db():.1f} dB")

    def _on_sat_released(self, _v=None):
        """松手后才真正应用。"""
        self._on_sat_label()
        self._apply()

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
        # 饱和起点只对线性幅度域有意义
        sat_on = self.cb_domain.currentData() == "linear"
        self.sld_sat.setEnabled(sat_on)
        self.lbl_sat_local.setEnabled(sat_on)

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
            "domain": self.cb_domain.currentData(),
            "channel_plane": self._current_channel_plane(),
            "channel_mode": self._channel_mode(),
            "sat_db": self._sat_db(),
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
        self.cb_domain.setCurrentIndex(max(0, self.cb_domain.findData("db")))
        self.cb_channel.setCurrentIndex(0)
        self.sld_sat.setValueSilently(self._sat_db_to_slider(DEFAULT_WT_SAT_DB))
        self.chk_beats.setChecked(True)
        self.sp_bpm.setValue(120.0)
        self.sp_bpb.setValue(4)
        self.sld_hide.setValueSilently(35)
        self._on_volume_label(35)
        self._on_sat_label()
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
