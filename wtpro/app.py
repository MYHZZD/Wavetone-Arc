"""主窗口与程序入口。"""

import math
import os
import sys
import tempfile

import numpy as np

from wtpro import backends
from wtpro import qt
from wtpro.audio import (
    AudioVariants, export_payload, export_result, load_audio, samples_to_wav_bytes
)
from wtpro.backends import (
    _backend_label, _backend_short, _ensure_backend_ready, _init_midi, _midi_label,
    _start_gpu_probe, midi_note_off, midi_note_on
)
from wtpro.color import DEFAULT_WT_SAT_DB
from wtpro.common import (
    CHORDS, CHORD_NONE, CUSTOM_CHORD_NAME, DEFAULT_CHANNEL_MODE,
    DEFAULT_ROWS_PER_SEMITONE, DEFAULT_TARGET_FPS, DEFAULT_WINDOW, MIDI_MAX, MIDI_MIN,
    MODE_PLANES, PLANE_LABELS, axis_samples, chord_intervals, midi_name, midi_to_freq,
    stereo_average
)
from wtpro.dialogs import (
    AnalysisParamsDialog, ChordBuilderDialog, FilterConfigDialog,
    SpectrumSettingsDialog
)
from wtpro.spectrogram import PianoRoll, SpectrogramView
from wtpro.worker import AnalysisWorker
from wtpro.qt import (
    HAS_MEDIA, QAction, QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QMediaContent, QMediaPlayer, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QStatusBar, QTimer, QToolBar, QUrl, QWidget, Qt, _notifier
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"频谱图分析工具  ·  {backends.GPU_NAME}")
        self.setMinimumSize(960, 640)
        self.resize(1400, 860)
        self.setAcceptDrops(True)

        self.samples = None
        self.sr = 44100
        self.hop = 512
        self.hide_low = 0.0
        self.current_path = None
        self.db = None          # 当前显示平面的 dB，供导出用
        self.mag = None         # 当前显示平面的线性幅度（未归一化）
        # 多声道：planes_db / planes_mag 按平面名存全部结果，
        # db / mag 始终指向"当前显示的那一张"，导出与其余代码只需认后者。
        self.planes_db = {}
        self.planes_mag = {}
        self.plane_ref = None

        self.params = {
            "window_name": DEFAULT_WINDOW,
            "rows_per_semitone": DEFAULT_ROWS_PER_SEMITONE,
            "target_fps": DEFAULT_TARGET_FPS,
            "interp": True,
            "channel_mode": DEFAULT_CHANNEL_MODE,
            "channel_l": False,
            "channel_r": False,
        }

        self.playback_samples = None
        # 当前出声的变体名（stereo | l | r），由 AudioVariants 决定。
        # 三个变体的播放器与临时 WAV 都由 AudioVariants 统一持有、统一释放。
        self._play_mode = None
        self._wav_buffer = None
        self._wav_temp_path = None

        self._is_playing = False
        self._worker = None

        # 这里原来有个 120 ms 的 _hide_timer，用来把"拖动中的过渡值"合并掉。
        # 现在滑块的 valueReleased 只在松手时发一次（ReleaseSlider），拖动中
        # 根本不触发处理，防抖已经是多余的；留着只会让松手后的画面晚 120 ms
        # 才更新，看着像"变了两次"。所以直接同步应用。
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

        self._audio = None
        self._init_media_player()

        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()

        # 后台探测完成后刷新标题栏/状态栏，否则界面会一直显示 CPU。
        _notifier.changed.connect(self._on_backend_changed)
        # 万一探测在界面建好之前就结束了，这里补一次。
        if backends._GPU_PROBE_DONE is not None and backends._GPU_PROBE_DONE.is_set():
            self._on_backend_changed(backends.GPU_STATUS)

        # 这里原来还有一句 QTimer.singleShot(0, resize(...))，但构造函数开头
        # 已经 resize 过一次、后面也没有依赖布局变化的尺寸调整，属于纯冗余，已删。

    def _on_backend_changed(self, name=""):
        """在主线程刷新与后端相关的文字。"""
        try:
            self.setWindowTitle(f"频谱图分析工具  ·  {backends.GPU_NAME}")
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
            self._audio = AudioVariants(self._new_player, self)
        except Exception as e:
            print(f"[media] 播放器管理初始化失败: {e}")
            self._audio = None

    def _new_player(self, parent=None):
        """按需创建一个播放器并接好信号。"""
        pl = QMediaPlayer(parent if parent is not None else self)
        try:
            pl.positionChanged.connect(self._on_player_position)
        except Exception:
            pass
        return pl

    @property
    def player(self):
        """当前出声的那个播放器。

        多播放器方案下"播放器"不再只有一个，所以收成属性：外部代码拿到的
        永远是当前出声的那一个，行为与以前一致。
        """
        aud = getattr(self, "_audio", None)
        if aud is None or not aud.active:
            return None
        ent = aud._entries.get(aud.active) or {}
        return ent.get("player")

    def _release_media(self):
        if self._audio is not None:
            try:
                self._audio.close()
            except Exception as e:
                print(f"[media] 释放播放器失败: {e}")
        if self._wav_buffer is not None:
            try:
                self._wav_buffer.close()
            except Exception:
                pass
            self._wav_buffer = None

    def _load_media_from_samples(self, samples, sr, source_path=None, mode="stereo"):
        """按分析模式预生成变体 WAV，并激活对应变体。

        **任何变体都必须编码成临时双声道 WAV**，不能让播放器直接吃源文件：
        Windows 上 QMediaPlayer 走 DirectShow，WAV 以外的格式会直接报
        `doRender: Unknown error 0x80040266`。多花一次 16 bit 编码换
        "什么格式都能放"，这个代价必须付。

        变体的取舍（用户选择的是"省盘"档）：
            stereo_only        -> 只生成 stereo
            l_only / r_only    -> 只生成对应那一路
            lr_both            -> stereo / l / r 三份都生成
        这样常见情况下只有 1 份临时文件；只有在"左+右"模式下才付 3 份
        （24 秒约 12.6 MB）。那份模式本来也要重新分析，所以代价落在同一次操作里。

        预生成的意义：之后切声道只是改音量，实测 0.01~0.06 ms，
        而重新 setMedia 装载每次要 ~50 ms、首次还有 ~1.2 s 的图重建。
        """
        if self._audio is None:
            return
        import tempfile

        # 变体集合必须**包含当前要激活的那一个**
        mode = mode if mode in ("l", "r") else "stereo"
        if mode == "stereo":
            wanted = ("stereo", "l", "r") if self._audio_wants_all() else ("stereo",)
        else:
            wanted = (mode,)

        self._release_media()
        for name in wanted:
            try:
                fd, tmp = tempfile.mkstemp(prefix="wavetonepro_", suffix=".wav")
                os.close(fd)
                if name == "stereo":
                    data = samples
                else:
                    data = axis_samples(samples, 0 if name == "l" else 1)
                AudioVariants.write_pcm_wav(tmp, data, sr)
            except Exception as e:
                print(f"[media] 变体 {name} 生成失败: {e}")
                continue
            self._audio.add(name, tmp)

        if not self._audio.names():
            self._wav_temp_path = None
            return
        # _wav_temp_path 只作展示/兜底用途：真正的清理由 AudioVariants.close()
        # 负责，它会把每个变体的临时文件都删掉（只删这一份是不够的）。
        self._wav_temp_path = self._audio._entries.get(mode, {}).get("path")
        # 把所有变体的播放器提前建好并暖一遍：不这么做，第一次切到某个变体
        # 要新建播放器 + DirectShow 建图，实测 ~450 ms。
        self._audio.prepare()
        self._audio.activate(mode)

    def _initial_audio_mode(self):
        """本次分析模式下，装载时要激活的变体。

        必须和分析模式的第一个平面一致：l_only 下只勾了左声道，要是这里
        还按 stereo 生成，就会出现"唯一能看的平面是 L、耳朵里却是立体声"，
        而且因为没生成 l 变体，切过去也切不动。
        """
        planes = self._plane_order()
        return self._plane_audio_mode(planes[0] if planes else "stereo")

    def _audio_wants_all(self):
        """当前分析模式是否值得预生成三个变体（即是否会产生 L / R 平面）。"""
        return self.params.get("channel_mode", DEFAULT_CHANNEL_MODE) == "lr_both"

    def _set_player_volume(self, value_0_1: float):
        """设置"当前出声变体"的音量（其余变体恒为静音）。"""
        if self._audio is None:
            return
        self._audio.set_volume(value_0_1)

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
        act_open.setToolTip(
            "打开音频文件并开始分析。\n"
            "快捷键：Ctrl+O")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_file)
        tb.addAction(act_open)

        act_params = QAction("参数 ⚙", self)
        act_params.setToolTip(
            "打开分析参数窗口，修改窗函数、纵轴与时间分辨率后重新分析。\n"
            "参数只作用于本次分析，修改后需重新分析才能生效。")
        act_params.triggered.connect(self.open_params_dialog)
        tb.addAction(act_params)

        self.act_export = QAction("导出 💾", self)
        self.act_export.setToolTip(
            "导出分析结果。**导出什么由你选的格式决定**：\n"
            "  .png —— 全分辨率频谱图，按当前显示设置（主题 / 显示域 / 感度或\n"
            "          亮度截断 / 阶梯滤镜 / 音域裁剪）渲染出来的那一张，\n"
            "          像素尺寸 = 帧数 × 当前显示的行数。\n"
            "  .npz —— 分析数据：mag（未归一化线性幅度，原始量纲）、\n"
            "          db（相对峰值，上限 0）、u8（当前显示设置下的色号），\n"
            "          外加 time_s / midi / freq_hz 坐标轴与全部分析参数。\n"
            "导出的都是**当前显示的那一张声道平面**，平面名写在元数据里。\n"
            "快捷键：Ctrl+E")
        self.act_export.setShortcut("Ctrl+E")
        self.act_export.triggered.connect(self.export_result_dialog)
        self.act_export.setEnabled(False)
        tb.addAction(self.act_export)

        act_fit = QAction("适应 🔍", self)
        act_fit.setToolTip(
            "缩放视图，使整段频谱在窗口内完整显示。\n"
            "快捷键：Ctrl+0")
        act_fit.setShortcut("Ctrl+0")
        act_fit.triggered.connect(lambda: self.spec.fit_view())
        tb.addAction(act_fit)

        tb.addSeparator()

        # 播放/暂停合成一个键：图标随状态变，点击即切换
        self.act_play = QAction("播放 ▶", self)
        self.act_play.setToolTip(
            "开始播放；播放过程中再次触发则暂停。\n"
            "快捷键：空格")
        self.act_play.setShortcut("Space")
        self.act_play.setShortcutContext(Qt.ApplicationShortcut)
        self.act_play.triggered.connect(self._on_play_pause)
        tb.addAction(self.act_play)

        self.act_stop = QAction("停止 ⏹", self)
        self.act_stop.setToolTip(
            "停止播放并将播放位置复位到音频开头。")
        self.act_stop.triggered.connect(self._on_stop)
        tb.addAction(self.act_stop)

        if self.player is None:
            self.act_play.setEnabled(False)
            self.act_stop.setEnabled(False)

        self.act_follow = QAction("跟随 🎯", self)
        self.act_follow.setCheckable(True)
        self.act_follow.setChecked(False)
        self.act_follow.setToolTip(
            "屏幕跟随播放头：播放时视图自动滚动，使播放位置保持在可见范围内。\n"
            "手动拖动频谱后该功能自动取消。")
        self.act_follow.toggled.connect(self._on_follow_toggled)
        tb.addAction(self.act_follow)

        tb.addSeparator()

        # 谱面设置：音域 / 外观 / 小节线 都收进这个窗口，工具栏只留一个入口
        self.act_spec_cfg = QAction("谱面设置 ⚙", self)
        self.act_spec_cfg.setToolTip(
            "打开谱面设置窗口，集中调整以下显示属性：\n"
            "  钢琴窗显示音域\n"
            "  主题配色 / 声道 / 显示域 / 感度\n"
            "  亮度截断\n"
            "  小节线与节拍线\n"
            "各项修改即时预览，点「确定」后生效。")
        self.act_spec_cfg.triggered.connect(self.open_spectrum_settings)
        tb.addAction(self.act_spec_cfg)

        tb.addSeparator()

        self.act_clear_mask = QAction("清空遮罩 🧹", self)
        self.act_clear_mask.setToolTip(
            "删除当前全部遮罩。\n"
            "该操作不可撤销。")
        self.act_clear_mask.triggered.connect(self._on_clear_masks)
        tb.addAction(self.act_clear_mask)

        # ---- 和弦 ----
        tb.addWidget(QLabel("和弦"))
        self.cb_chord = QComboBox()
        self.cb_chord.setToolTip(
            "和弦模式：以鼠标所在音高作为低音，一次标记、遮罩或试听多个音。\n"
            "音程自低音向上计算，因此改变转位不会移动低音位置。\n"
            "选择「关闭」则恢复单音模式。")
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
            "和弦转位：把最低的 n 个音依次升高一个八度。\n"
            "低音位置保持不变，仅改变和弦的排列方式。")
        self.sp_chord_inv.valueChanged.connect(self._on_chord_changed)
        tb.addWidget(self.sp_chord_inv)

        self.act_chord_custom = QAction("自定义和弦 🎹", self)
        self.act_chord_custom.setToolTip(
            "自定义和弦：打开覆盖两个八度的钢琴窗，逐音点选所需的和弦形状。")
        self.act_chord_custom.setCheckable(True)
        self.act_chord_custom.toggled.connect(self._on_chord_custom_toggled)
        tb.addAction(self.act_chord_custom)

        # 泛音数量：它的效果要靠鼠标悬停看，放对话框里没法预览，所以留在工具栏
        tb.addWidget(QLabel("泛音"))
        self.cb_harm = QComboBox()
        self.cb_harm.setToolTip(
            "泛音高亮数量：鼠标悬停时，自基音向上依次点亮的泛音个数。\n"
            "0 表示仅显示基音。\n"
            "超出当前显示音域上限的泛音不予绘制。")
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
            "阶梯滤镜：把每个半音内的各行归约为同一数值，使半音内部均匀、\n"
            "半音之间出现明显的分界（阶梯效果）。\n"
            "该滤镜仅作用于显示，不改变分析结果与导出数据。\n"
            "归约方式与对比曲线在「阶梯参数」中设置。")
        self.act_step.toggled.connect(self._on_step_filter_toggled)
        tb.addAction(self.act_step)

        self.act_step_cfg = QAction("阶梯参数 ⚙", self)
        self.act_step_cfg.setToolTip(
            "打开阶梯滤镜参数窗口，设置归约方式、去边行数与对比曲线。\n"
            "窗口内提供实时预览。")
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
    # 声道平面
    # ------------------------------------------------------------------
    def _plane_order(self):
        """当前分析模式下，平面的优先顺序（stereo → l → r）。"""
        mode = self.params.get("channel_mode", DEFAULT_CHANNEL_MODE)
        return MODE_PLANES.get(mode, ("stereo",))

    def active_plane(self):
        """当前正在显示的平面名。

        优先用频谱图控件上记录的那个；还没设过就按"有 stereo 用 stereo，
        没有就用第一个候选"（也就是只有 L 时显示 L）。
        """
        cur = getattr(self.spec, "channel_plane", None)
        if cur and cur in self.planes_db:
            return cur
        for p in self._plane_order():
            if p in self.planes_db:
                return p
        return next(iter(self.planes_db), "stereo")

    def _plane_audio_mode(self, plane):
        """平面名 → 播放源模式。

        stereo 放完整的立体声；l / r 放单侧复制成双声道的等响度单声道。
        两种模式都必须编码成临时 WAV（见 _load_media_from_samples）。
        """
        return plane if plane in ("l", "r") else "stereo"

    def _restore_playback(self, plane=None):
        """按当前显示的平面切换正在播放的变体。

        不重新生成 WAV、不碰 setMedia：变体在装载文件时就一次生成好、播放器
        也提前建好并暖过了，切换只是"暂停旧的、seek 新的、播新的"，
        实测 0.1 ms 量级，播放中热切换同样成立。
        """
        if self._audio is None or not self._audio.names():
            return
        mode = self._plane_audio_mode(plane or self.active_plane())
        if not self._audio.has(mode):
            # 当前分析模式没有预生成这一路（例如 stereo_only 下只有 stereo）。
            # 退回任一可用变体，不为了听一路声音去重算整段音频。
            mode = self._audio.active or self._audio.names()[0]
        self._play_mode = mode
        self._audio.activate(mode)

    def _set_active_plane(self, plane):
        """切换显示的平面：只换数据不重算，视图与交互状态全部保留。

        只是换了一张同形状的图，没有理由让视口跳回开头、也不该丢掉
        跟随模式、遮罩和手动拖动状态 —— set_data 会重置这些，所以先存后还原。
        """
        if plane not in self.planes_db:
            return
        db = self.planes_db[plane]
        mag = self.planes_mag.get(plane)
        self.db = db
        self.mag = mag

        # 同形状换图，下面这些状态都应当保持原样
        keep = {}
        for attr in ("view_start", "scale", "playhead_frame", "_follow_ratio",
                     "_has_manual_seek", "selected_mask", "masks"):
            if hasattr(self.spec, attr):
                keep[attr] = getattr(self.spec, attr)

        # plane 必须传给 set_data：上色缓存的指纹里有平面名，
        # 否则切回来时会命中上一张平面的缓存。
        # invalidate_cache=False：只是换张图，别把缓存清了 —— 这正是切声道
        # 之所以卡的原因（每张图重新上色 + 重建 QImage 约 425 ms）。
        self.spec.set_data(db, self.hop, self.sr, fit=False,
                           rows_per_semitone=self.params["rows_per_semitone"],
                           plane=plane, invalidate_cache=False)

        for attr, val in keep.items():
            try:
                # 列表类状态按副本还原，避免与控件内部持有的同一个对象互相污染
                setattr(self.spec, attr, list(val) if isinstance(val, list) else val)
            except (AttributeError, TypeError):
                pass
        try:
            self.spec._clamp_view()
        except (AttributeError, TypeError, ValueError):
            pass
        self.spec._invalidate()

        self._restore_playback(plane)
        self.spec.update()

    def _announce_plane(self, planes_db, hop):
        """状态栏文字：帧数行数 + 当前平面 + 本次算了几个平面。"""
        db = self.planes_db.get(self.active_plane())
        if db is None:
            return
        n_frames, n_rows = db.shape
        backend = _backend_short()
        mode = self.params.get("channel_mode", DEFAULT_CHANNEL_MODE)
        runs = 2 if mode == "lr_both" else 1
        names = " / ".join(PLANE_LABELS.get(p, p) for p in self._plane_order()
                           if p in self.planes_db)
        self.lbl_left.setText(
            f"就绪  ·  {backend}  ·  {n_frames} 帧 × {n_rows} 行  ·  hop {hop} "
            f"({self.sr / hop:.1f} fps)  ·  {self.params['window_name']}  ·  "
            f"{self.params['rows_per_semitone']} 行/半音  ·  "
            f"显示 {PLANE_LABELS.get(self.active_plane(), self.active_plane())}"
            f"（算 {runs} 次，{names}）")

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
            "domain": getattr(self.spec, "domain", "db"),
            "sat_db": float(getattr(self.spec, "sat_db", DEFAULT_WT_SAT_DB)),
            # 声道：显示哪一张平面，以及"可选项由哪种分析模式决定"
            "channel_plane": self.active_plane(),
            "channel_mode": self.params.get("channel_mode", DEFAULT_CHANNEL_MODE),
            "show_beats": bool(self.spec.show_beats),
            "bpm": float(self.spec.bpm),
            "beats_per_bar": int(self.spec.beats_per_bar),
            "hide_low": float(self.hide_low),
            "volume_percent": int(self._volume_percent),
        }

    def _apply_spectrum_settings(self, s):
        """把一份设置推到各个视图上。窗口里改一处就会走一次。

        亮度截断的**阈值**由窗口按当前显示域算好放在 s["hide_low"] 里
        （两个域的映射区间不同，见 SpectrumSettingsDialog.TRUNC_RANGE），
        这里只负责落到频谱图上；volume_percent 仅用于同步对话框滑块位置。
        """
        lo = int(s.get("midi_min", MIDI_MIN))
        hi = int(s.get("midi_max", MIDI_MAX))
        if (lo, hi) != (self.spec.midi_min, self.spec.midi_max):
            self.piano.set_midi_range(lo, hi)
            self.spec.set_midi_range(lo, hi)
        plane = s.get("channel_plane")
        if plane and plane != self.active_plane() and plane in self.planes_db:
            self._set_active_plane(plane)
        self.spec.set_colormap(s.get("colormap", self.spec.cmap))
        if hasattr(self.spec, "set_domain"):
            self.spec.set_domain(s.get("domain", getattr(self.spec, "domain", "db")))
        if hasattr(self.spec, "set_sat_db"):
            self.spec.set_sat_db(s.get("sat_db",
                                       getattr(self.spec, "sat_db", DEFAULT_WT_SAT_DB)))
        self.spec.set_bpm(float(s.get("bpm", self.spec.bpm)))
        self.spec.set_beats_per_bar(int(s.get("beats_per_bar", self.spec.beats_per_bar)))
        self.spec.set_show_beats(bool(s.get("show_beats", self.spec.show_beats)))
        # 亮度截断阈值：以窗口算好的为准，别再拿百分比现算。立即生效 ——
        # 滑块松手才发信号，所以"过渡值"根本不会进来。
        if s.get("hide_low") is not None:
            self.hide_low = float(s["hide_low"])
            self._apply_hide_low()
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
        ok = False
        final_settings = None
        try:
            ok = dlg.exec_() == QDialog.Accepted
            # 必须在 finally 的 deleteLater 之前取值：销毁排程之后 dlg 的
            # 子控件已经不可靠，get_settings() 会读到已回收的窗口部件。
            if ok:
                final_settings = dlg.get_settings()
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
        if ok and final_settings is not None:
            self._apply_spectrum_settings(final_settings)
        self.spec.setFocus()

    def _on_spec_clicked(self, frame_pos, midi_note):
        self.spec.set_playhead_frame(frame_pos)
        if self._audio is not None and self._audio.names() and self.hop > 0 and self.sr > 0:
            ms = int(frame_pos * self.hop * 1000.0 / self.sr)
            try:
                # 三个变体一起跳：只跳出声的那一个，切声道后就会跳到别处去。
                self._audio.set_position(ms)
            except Exception as e:
                print(f"[media] setPosition 失败: {e}")
            self.lbl_left.setText(f"跳转至  {frame_pos * self.hop / self.sr:.3f} s   ·   " f"{midi_name(midi_note)}")

    def _play_midi_note(self, midi):
        """和弦模式下一次点击把和弦的所有音一起发出去。"""
        if not backends.MIDI_AVAILABLE:
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
        """滑块松手（ReleaseSlider 只在松手时发 valueReleased）。

        现在这条直连信号**只用于同步显示值**：阈值由「谱面设置」按当前显示域
        算好、随 _apply_spectrum_settings 一起送过来并立即生效。窗口自己会走
        同一条路，所以这里不再重复触发应用（重复只会多算一次）。
        """
        self._volume_percent = int(v)

    def _apply_hide_low(self):
        """把当前阈值推给频谱图。

        阈值本身由「谱面设置」窗口算好（它才知道当前显示域对应哪一段区间），
        这里读缓存的 hide_low，不再拿百分比现算 —— 那样会把线性域的映射算错。
        线性域里 spec.set_hide_low 只记录不重绘（那个域不参与截断）。
        """
        if abs(self.hide_low - self.spec.hide_low) < 1e-6:
            return
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
        if self._audio is None or self.current_path is None:
            return
        if not self._audio.names():
            return
        try:
            self._audio.play()
        except Exception as e:
            print(f"[media] play 失败: {e}")
            return
        self._is_playing = True
        self._set_play_icon(True)
        try:
            self._apply_position_ms(self._audio.position())
        except Exception:
            pass
        self._playhead_timer.start()

    def _on_pause(self):
        if self._audio is None:
            return
        try:
            self._audio.pause()
        except Exception as e:
            print(f"[media] pause 失败: {e}")
        self._is_playing = False
        self._set_play_icon(False)
        self._playhead_timer.stop()

    def _on_stop(self):
        if self._audio is None:
            return
        try:
            # 只归零、不释放：三个变体的 WAV 与播放器都留着，下次播放/切声道
            # 依然是零延迟。
            self._audio.stop()
        except Exception as e:
            print(f"[media] stop(soft) 失败: {e}")
        self._is_playing = False
        self._set_play_icon(False)
        self._playhead_timer.stop()
        self.spec.reset_follow_state()
        self.spec.set_playhead_frame(0.0)
        self.lbl_left.setText("已停止，播放头归零")

    def _tick_playhead(self):
        if self._audio is None or not self._is_playing:
            return
        try:
            ms = self._audio.position()
        except Exception:
            return
        self._apply_position_ms(ms)

    def _on_player_position(self, ms):
        """位置回调。

        只有**当前出声**的那个播放器的位置能驱动播放头：静音变体也在同步
        播放并各自发位置信号，不挡掉就会几个源互相打架、播放头来回跳。
        """
        if self._audio is None or not self._is_playing:
            return
        sender = self.sender()
        if sender is not None and sender is not self.player:
            return
        self._apply_position_ms(ms)

    def _apply_position_ms(self, ms):
        if self.hop <= 0 or self.sr <= 0:
            return
        frame = (ms / 1000.0) * self.sr / float(self.hop)
        self.spec.set_playhead_frame(frame)

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

        # 分析在 worker 里按声道模式取声道，所以这里**保留原始的立体声数据**，
        # 不再事先折叠成单声道 —— 折叠过一次，L / R 平面就没法还原了。
        samples = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
        self.samples = samples
        self.sr = sr
        self.current_path = path
        self._is_playing = False

        self.playback_samples = samples
        # 预生成变体 WAV 并激活默认那一路。必须在 _rebuild 之前做完：
        # 之后切声道就只是改音量，不再有任何文件操作。
        try:
            self._load_media_from_samples(samples, sr, source_path=path,
                                          mode=self._initial_audio_mode())
        except Exception as e:
            print(f"[media] 播放源准备失败: {e}")
        self._play_mode = self._audio.active if self._audio else None

        name = os.path.basename(path)
        dur = len(samples) / float(sr) if samples.ndim == 1 else samples.shape[0] / float(sr)
        ch_txt = ""
        try:
            s = np.asarray(samples)
            if s.ndim == 2:
                ch_txt = f"   ·   {s.shape[1]}ch"
                if s.shape[1] == 1:
                    ch_txt += "（单声道，L/R 等价）"
            else:
                ch_txt = "   ·   1ch（单声道，L/R 等价）"
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
            # 只要有变体可放就允许播放；此刻可能还没创建播放器实例
            # （播放器是真正开始播时才建的），所以不能拿 self.player 判断。
            if self._audio is not None and self._audio.names():
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

    def _on_analysis_done(self, planes_db, hop, planes_mag=None, ref=None):
        """接收多平面结果。

        进入前先把 db / mag 指向默认要显示的那一张（有 stereo 就 stereo，
        否则第一个候选 —— 只有 L 时就是 L），再走一次正常的 set_data。
        """
        self._set_busy(False)
        self.hop = hop
        self.planes_db = dict(planes_db or {})
        self.planes_mag = dict(planes_mag or {})
        self.plane_ref = ref

        plane = self._plane_order()[0]
        if plane not in self.planes_db:
            plane = next(iter(self.planes_db), "stereo")
        self.db = self.planes_db.get(plane)
        self.mag = self.planes_mag.get(plane)

        if self.db is None:
            self.lbl_left.setText("分析完成，但没有产出任何声道平面")
            return

        self.spec.hop = hop
        self.spec.sr = self.sr
        # 新一批结果：缓存必须作废（invalidate_cache 默认 True），
        # plane 一并告诉控件，缓存的指纹才落在这张平面上。
        self.spec.set_data(self.db, hop, self.sr, fit=True,
                           rows_per_semitone=self.params["rows_per_semitone"],
                           plane=plane)
        self.spec.set_playhead_frame(0.0)
        try:
            self.act_export.setEnabled(True)
        except Exception:
            pass

        self._restore_playback(plane)
        self._announce_plane(self.planes_db, hop)
        self.lbl_center.setText("")
        self.spec.setFocus()
        # 其余平面延后到事件循环里预热：每张图上色+建 QImage 约 450 ms
        # （20 000 × 1056），三张近 1.4 s，压在这里会让"分析完成"之后卡一下。
        # 放到 0 ms 定时器里既能让界面先刷出来，又能在用户去点声道下拉框之前
        # 备好缓存 —— 那之后切声道就是 0.02 ms 级。
        self._prime_plane_cache()

    def _prime_plane_cache(self):
        """把还没上色过的平面逐张预涂，避免首次切过去时卡一下。"""
        pending = [p for p in self._plane_order()
                   if p in self.planes_db and p != self.active_plane()]

        def step():
            if not pending:
                return
            name = pending.pop(0)
            try:
                self.spec.prime_plane(name, self.planes_db[name], self.hop, self.sr,
                                      rows_per_semitone=self.params["rows_per_semitone"])
            except Exception as e:
                print(f"[display] 预涂平面 {name} 失败: {e}")
                return
            if pending:
                QTimer.singleShot(0, step)

        if pending:
            QTimer.singleShot(0, step)

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
    # 导出（数据 / 渲染图）
    # ------------------------------------------------------------------
    def export_result_dialog(self):
        """导出分析结果。**导出什么由选的格式决定**：

            npz  数据 —— mag / db / u8 + 坐标轴 + 全部元数据
            png  全分辨率频谱图 —— 按当前显示设置渲染出来的那张图
        """
        if self.db is None:
            QMessageBox.information(self, "无可导出数据", "请先打开并分析一个音频文件。")
            return

        base = "analysis"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
        start = os.path.join(os.path.dirname(self.current_path or ""), base + ".npz")

        # 用完整的 QFileDialog 而不是 getSaveFileName()，为的是解决一个 Qt 的老毛病：
        # getSaveFileName 只返回"用户输入的字符串 + 选中哪个筛选器"，
        # **切换筛选器时它不会动文件名里的后缀**。于是"选了 png、文件名还是 .npz"，
        # 用户拿到一个后缀和内容不一致的文件。
        #
        # 双保险：
        #   * filterSelected  —— 用户在对话框里切换筛选器时，当场把文件名后缀换掉，
        #                        所见即所得。实测程序化 selectNameFilter 不触发它，
        #                        所以它只管"用户在操作"这一路。
        #   * 接受后归位      —— 不论中途发生什么，最终按**筛选器**把后缀换成
        #                        正确的那一个。这一步是权威，不能省。
        # 注意是**替换**后缀不是追加：/a/song.npz 选 png 要得到 /a/song.png，
        # 而不是 /a/song.npz.png。
        flt_png = "全分辨率频谱图 (*.png)"
        flt_npz = "分析数据 (*.npz)"
        dlg = QFileDialog(self, "导出（格式决定内容）", start,
                          f"{flt_png};;{flt_npz}")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setDefaultSuffix("png")          # 列表第一项，也是默认选项

        def _as_ext(path, ext):
            """把 path 的后缀换成 ext（没有后缀就补上）。"""
            return os.path.splitext(path)[0] + ext

        def _sync_suffix(selected):
            ext = ".png" if "png" in str(selected) else ".npz"
            dlg.setDefaultSuffix(ext.lstrip("."))
            cur = dlg.selectedFiles()
            if cur and os.path.splitext(cur[0])[1].lower() != ext:
                dlg.selectFile(_as_ext(cur[0], ext))

        dlg.filterSelected.connect(_sync_suffix)

        if dlg.exec_() != QFileDialog.Accepted:
            return
        picked = dlg.selectedFiles()
        if not picked:
            return
        path = picked[0]
        # 筛选器是权威：用户明确选了哪种格式就导出哪种，后缀跟着它归位。
        fmt = "png" if "png" in str(dlg.selectedNameFilter()) else "npz"
        path = _as_ext(path, ".png" if fmt == "png" else ".npz")

        # ---- PNG：用视图里已经上色好的那张图，不重算上色 ----
        if fmt == "png":
            # 注意用 export_qimage() 而不是 spec.qimg：
            # 开了阶梯滤镜时 u8/qimg 是"一个半音一行"的**紧凑**形式（屏幕靠
            # QPainter 纵向拉伸显示）。直接存紧凑形式，图高就只有半音数
            # （88 而不是 1056），看着像"导出丢了行"。
            # export_qimage() 先把紧凑行按 rows_per_semitone 复制回真实行数，
            # 再按音域裁剪，得到与屏幕上完全一致的全分辨率结果。
            get_img = getattr(self.spec, "export_qimage", None)
            img = get_img() if callable(get_img) else getattr(self.spec, "qimg", None)
            if img is None or img.isNull():
                QMessageBox.information(
                    self, "无可导出画面",
                    "频谱图还没渲染出来（可能窗口尚未显示）。\n"
                    "请先让主窗口把图画出来再导出。")
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                out = export_result(path, None, fmt, img=img)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "导出失败", f"{type(e).__name__}: {e}")
                return
            QApplication.restoreOverrideCursor()
            h, w = img.height(), img.width()
            # 把"每个半音占几行"写出来：图的像素高度直接由它决定，
            # 用户看到"怎么只有 88 像素高"时，答案就在这里。
            rps = int(self.params.get("rows_per_semitone",
                                      DEFAULT_ROWS_PER_SEMITONE))
            rows_note = (f"{h} 行（{rps} 行/半音）" if rps > 1
                         else f"{h} 行（1 行/半音，即一个半音一行）")
            self.lbl_left.setText(
                f"已导出 {os.path.basename(out)}   ·   PNG   ·   "
                f"{w} × {h} px = {w} 帧 × {rows_note}   ·   "
                f"{self._file_size_text(out)}   ·   "
                f"{midi_name(self.spec.midi_min)}–{midi_name(self.spec.midi_max)}")
            # 分析分辨率低的时候提示一句，免得以为导出坏了
            if rps <= 1:
                self.lbl_center.setText(
                    "提示：本次分析是 1 行/半音，图高即半音数；"
                    "要更细的纵向分辨率请在「分析参数」里提高行/半音后重新分析")
            else:
                self.lbl_center.setText("")
            return

        # ---- NPZ：数据 ----
        # mag 是未归一化的线性幅度（原始数据）；u8 是当前显示设置下的色号。
        # 导出的是**当前显示的那一张平面**，把它的名字一并写进元数据。
        mag = self.mag
        if mag is None:
            # 兜底：从 dB 反推（会受 floor 裁剪影响，仅用于老数据）
            mag = np.power(np.float64(10.0),
                           np.asarray(self.db, dtype=np.float64) / 20.0).astype(np.float32)
        exp_params = dict(self.params)
        exp_params["channel_plane"] = self.active_plane()
        payload = export_payload(
            mag, self.db, self.hop, self.sr, params=exp_params,
            source_path=self.current_path, hide_low=self.hide_low,
            colormap=getattr(self.spec, "cmap", None),
            u8=getattr(self.spec, "u8", None),
            domain=getattr(self.spec, "domain", None),
            sat_db=getattr(self.spec, "sat_db", None),
        )

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            out = export_result(path, payload, fmt)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "导出失败", f"{type(e).__name__}: {e}")
            return
        QApplication.restoreOverrideCursor()

        n_frames, n_rows = self.db.shape
        self.lbl_left.setText(
            f"已导出 {os.path.basename(out)}   ·   NPZ   ·   "
            f"{n_frames} 帧 × {n_rows} 行   ·   {self._file_size_text(out)}")
        self.lbl_center.setText("")

    @staticmethod
    def _file_size_text(path):
        """把字节数写成人看的大小。取不到就返回 '?'。"""
        try:
            size = os.path.getsize(path)
        except OSError:
            return "?"
        return f"{size/1048576:.1f} MB" if size >= 1048576 else f"{size/1024:.0f} KB"

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
        # AudioVariants.close() 会停掉并销毁全部变体播放器，并删掉它们的临时
        # WAV（三份都要删，不能只删 _wav_temp_path 记着的那一份）。
        self._release_media()
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
    if backends.MIDI_AVAILABLE:
        print(f"[midi] 已连接输出: {backends.MIDI_PORT_NAME}  (backend={backends.MIDI_BACKEND})")
    else:
        print("[midi] 未检测到可用输出（点击频谱不会发声）")

    win = MainWindow()
    win.show()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win.load_path(sys.argv[1])

    sys.exit(app.exec_())


