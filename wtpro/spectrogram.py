"""频谱图视图与钢琴卷帘控件。"""

import math

import numpy as np

from wtpro import backends
from wtpro.color import (
    LUTS, LUTS_RGB, WT_SAT_DB_MAX, WT_SAT_DB_MIN, wt_default_linear_sat_db
)
from wtpro.common import (
    BLACK_PC, DB_FLOOR, DEFAULT_ROWS_PER_SEMITONE, DISPLAY_GAMMA, MASK_MIN_DRAG_PX,
    MIDI_MAX, MIDI_MIN, _num_opt, chord_note_set, midi_name, midi_to_freq, midi_to_y,
    y_to_midi
)
from wtpro.render import (
    DEFAULT_FILTER_CONFIG, DOMAIN_MODES, _filter_config, db_to_u8, scaled_trim,
    semitone_bands, semitone_step_filter, u8_to_qimage_fast
)
from wtpro.qt import (
    QColor, QFont, QPainter, QPen, QPixmap, QPointF, QPolygonF, QRectF, QSize,
    QSizePolicy, QTimer, QWidget, Qt, pyqtSignal
)

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
        self.domain = "db"                      # db | linear（linear 复刻 WaveTone）
        self.sat_db = wt_default_linear_sat_db()   # 线性域的感度（dB 刻度）
        # 当前显示哪一张声道平面（stereo | l | r）。数据由主窗口在切换时
        # 通过 set_data 推进来，这里只记录"现在看的是哪一张"，供状态栏与
        # 谱面设置窗口读回。
        self.channel_plane = "stereo"
        self.show_grid = True

        # 阶梯滤镜：半音内取平均，只影响显示
        self.rows_per_semitone = DEFAULT_ROWS_PER_SEMITONE
        self.step_filter = False
        self.step_config = dict(DEFAULT_FILTER_CONFIG)
        self._step_cache = None
        self._trim_user_set = False  # 用户是否手动改过 midmax 的 N

        # 按 (平面, 显示参数) 缓存"上色后的 u8 + 裁好的 QImage"。
        #
        # 为什么必须缓存：切声道只是换一张已经算好的图，但上色 + 重建 QImage
        # 是纯 CPU 的整幅重算。实测 20 000 × 1056 的一张开销约 425 ms
        # （db_to_u8 97 ms + u8_to_qimage_fast 328 ms），来回切两下就是近 1 秒，
        # 表现成"明明图早就画完了，切声道却很卡"。
        # 缓存命中时切平面是零成本的（只剩一次 drawImage）。
        self._u8_cache = {}
        self._u8_cache_order = []

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

    def set_data(self, db, hop, sr, fit=True, rows_per_semitone=None, plane=None,
                 invalidate_cache=True):
        """装载一帧分析结果。

        plane 是这张图对应的声道平面名（stereo | l | r）。带缓存换图时它必须
        传进来，否则上色缓存的指纹会错位到上一张平面上。

        invalidate_cache 默认 True（换了一批新数据，旧 u8/QImage 全部作废）。
        只是切换到另一张**同一批结果里**的平面时传 False —— 那正是缓存的用武
        之地，清掉就等于每次切声道都白算一遍整幅图。
        """
        self.db = db
        if plane:
            self.channel_plane = str(plane)
        self.hop = hop
        self.sr = sr
        self.n_frames, self.n_rows = db.shape
        if rows_per_semitone is not None and int(rows_per_semitone) != self.rows_per_semitone:
            old_rps = self.rows_per_semitone
            self.rows_per_semitone = int(rows_per_semitone)
            self._rescale_trim(old_rps)
        self._step_cache = None
        if invalidate_cache:
            # 换了数据源，之前的 u8 / QImage 全部作废（同一份平面也可能因为
            # 重新分析而变），不清就会拿旧图当新图显示。
            self._u8_cache.clear()
            self._u8_cache_order.clear()
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
        """切换配色主题。

        两处与性能有关的讲究：
          1. 主题没变就直接返回。对话框点"确定"时会把整套设置重放一遍，
             原来这里无条件重建整张 QImage（20 000 × 1056 实测 333 ms），
             于是"预览时很流畅、一点确定就卡一下"。
          2. 主题变了走 _regen_u8 而不是直接 _rebuild_qimg：_regen_u8 认
             上色缓存（缓存键里有主题），切回上一个主题时能直接命中，
             并且会顺手把新的这份存进去。直接调 _rebuild_qimg 等于绕过缓存，
             每次都要从 u8 重新铺一遍像素。
        """
        if name not in LUTS or name == self.cmap:
            return
        self.cmap = name
        self.lut = LUTS[name]
        self.rgb_lut = LUTS_RGB[name]
        self._regen_u8()
        self._invalidate_bg()

    def set_domain(self, domain):
        """切换 dB / 线性显示域。linear 即复刻 WaveTone 的线性幅度显示。"""
        domain = str(domain)
        if domain not in DOMAIN_MODES:
            domain = "db"
        if domain == self.domain:
            return
        self.domain = domain
        self._regen_u8()
        self._invalidate_bg()

    def set_sat_db(self, sat_db):
        """设置线性域的感度（以 dB 刻度表示）。不改数据，只重算上色。"""
        default = wt_default_linear_sat_db()
        sat = _num_opt(sat_db, default)
        if not math.isfinite(sat):
            sat = default
        sat = max(WT_SAT_DB_MIN, min(WT_SAT_DB_MAX, sat))
        if abs(sat - self.sat_db) < 1e-9:
            return
        self.sat_db = sat
        if self.domain == "linear":
            self._regen_u8()
            self._invalidate_bg()

    def _rebuild_qimg(self):
        """按当前显示的音域裁出 QImage。

        只保留 [midi_min, midi_max] 对应的那些行，于是隐藏底部/顶部半音时
        中间区域会自动被拉高，其余绘制逻辑一行都不用改 —— 它们都是通过
        midi_to_y / y_to_midi 用 midi_min/midi_max 换算的。
        """
        img = self._full_qimg()
        if img is None:
            self.qimg = None
            return
        span = self._full_span()
        if span is None:
            self.qimg = img
            return
        lo, hi = span
        if lo == 0 and hi == img.height():
            self.qimg = img
            return
        self.qimg = img.copy(0, lo, img.width(), max(1, hi - lo))

    def _expanded_u8(self):
        """把**紧凑**的 u8 纵向复制回"真实行数"。

        阶梯滤镜开启时 `_display_db` 会走紧凑路径，u8 是"一个半音一行"
        （n_rows // rows_per_semitone 行）。屏幕上没问题 —— QPainter 会把
        每一行纵向拉满，肉眼看到的就是每半音占 rows_per_semitone 行。

        但**导出 PNG 必须还原**：紧凑形式直接存成图，高度就只有半音数
        （88 而不是 1056），用户看到的是"导出丢了行"。这里按 rows_per_semitone
        逐行复制，得到与屏幕上完全一致的全分辨率结果。

        没开阶梯滤镜（或没走紧凑路径）时原样返回。
        """
        if self.u8 is None:
            return None
        if not self._compact_bands():
            return self.u8
        rps = max(1, int(self.rows_per_semitone))
        h = self.u8.shape[1]
        used = h * rps
        if used >= self.n_rows:
            # 正好（或已超过）：直接铺开，末尾多余的行由下面的裁剪切掉
            return np.repeat(self.u8, rps, axis=1)[:, :self.n_rows]
        # 末尾不足一个半音的行：按最后一行的值补齐，与 semitone_step_filter
        # 的收尾方式一致
        out = np.empty((self.u8.shape[0], self.n_rows), dtype=self.u8.dtype)
        out[:, :used] = np.repeat(self.u8, rps, axis=1)
        out[:, used:] = self.u8[:, -1:]
        return out

    def _full_qimg(self):
        """未经音域裁剪的 QImage。展开后的高度与 semitone_step_filter 的
        "铺回 n_rows" 完全一致（同样的 used / 收尾规则）。"""
        u8 = self._expanded_u8()
        if u8 is None:
            return None
        return u8_to_qimage_fast(np.ascontiguousarray(u8), self.rgb_lut)

    def _full_span(self):
        """在**展开后**的高度坐标系里的可见行区间 [lo, hi)。

        与 `_visible_row_span` 的区别：这里一律按真实行数（每半音
        rows_per_semitone 行）算，因为展开后的图就是那个高度。
        返回 None 表示不需要裁剪（正好是全高）。
        """
        if self.u8 is None:
            return None
        rps = max(1, int(self.rows_per_semitone))
        h = max(1, int(self.n_rows))
        top = MIDI_MAX - int(self.midi_max)
        bottom = MIDI_MAX - int(self.midi_min) + 1
        lo = max(0, min(h, top * rps))
        hi = max(lo + 1, min(h, bottom * rps))
        if lo == 0 and hi == h:
            return None
        return lo, hi

    def export_qimage(self):
        """给导出用的全分辨率渲染图：阶梯滤镜的紧凑行已复制回真实行数，
        并按当前音域左右上下裁剪到可见范围。"""
        img = self._full_qimg()
        if img is None:
            return None
        span = self._full_span()
        if span is None:
            return img
        lo, hi = span
        if lo == 0 and hi == img.height():
            return img
        return img.copy(0, lo, img.width(), max(1, hi - lo))

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
        """设置亮度截断阈值。

        **只有 dB 域才需要重算显示**：线性域里这个阈值不参与映射（那个滑块在
        线性域控制的是感度），可是它仍然会被赋值/清零。如果照旧无条件重算，就会
        出现"用户改的是感度、画面却在一个延迟之后又跳一下"——那次跳变是这里
        重算出来的，值本身没变。所以线性域只记下数值、等切回 dB 域时再用。
        """
        v = float(v)
        if abs(v - self.hide_low) < 1e-6:
            return
        self.hide_low = v
        if self.db is not None and self.domain != "linear":
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

    def _display_key(self):
        """影响"上色结果"的全部显示参数的指纹。

        只要这个指纹不变、平面不变，同一份 db 上色出来的 u8 与 QImage 就一定
        一样，可以直接复用。改主题、显示域、饱和起点、亮度滤镜、阶梯滤镜、
        音域或每半音行数都会换掉指纹，缓存自然失效。
        """
        try:
            step_cfg = tuple(sorted((str(k), repr(v))
                                    for k, v in (self.step_config or {}).items()))
        except Exception:
            step_cfg = ()
        return (
            str(self.channel_plane),
            str(self.cmap),
            str(self.domain),
            float(self.sat_db) if self.sat_db is not None else None,
            float(self.hide_low),
            bool(self.step_filter),
            step_cfg,
            int(self.rows_per_semitone),
            int(self.midi_min),
            int(self.midi_max),
        )

    def _regen_u8(self):
        """把当前 dB 上色成 u8 并生成 QImage（带按平面缓存）。

        阶梯滤镜开启时先压成"一个半音一行"再上色，省掉 np.repeat 铺开和
        随之而来的 rps 倍重复上色 —— 这是滤镜路径最大的一笔开销。

        缓存键包含平面名与全部显示参数（见 _display_key），所以：
          * 切声道再切回来 -> 命中，不再重算；
          * 改主题/显示域/亮度等 -> 指纹变了，重算并留下新的那一份。
        """
        if self.db is None:
            self.u8 = None
            self.qimg = None
            return

        key = self._display_key()
        cached = self._u8_cache.get(key)
        if cached is not None:
            self.u8, self.qimg = cached
            return

        src = self._display_db()
        self.u8 = db_to_u8(src, floor_db=DB_FLOOR,
                           hide_low=self.hide_low, gamma=DISPLAY_GAMMA,
                           domain=self.domain, sat_db=self.sat_db)
        self._rebuild_qimg()
        self._remember_u8(key)
        # 注意：这里**不能**调 _trim_u8_cache()。换平面本身就会走这条路，
        # 一收敛就把刚缓存好的其他平面全丢了，等于永远不命中。
        # 收敛只发生在"显示参数变了、旧指纹再也用不到"的地方，见下面各 setter。

    # ------------------------------------------------------------------
    # 上色缓存
    # ------------------------------------------------------------------
    # 同时保留 3 份（stereo / l / r 正好各一份）。再多的历史平面缓存意义不大，
    # 而每份 QImage 是 n_frames × n_rows × 3 字节（20 000 帧约 63 MB），
    # 不设上限会在长时间来回切换时把内存吃满。

    _U8_CACHE_KEEP = 3

    def prime_plane(self, plane, db, hop, sr, rows_per_semitone=None):
        """预先把某个平面上色好塞进缓存，**不改变当前显示的内容**。

        每张图上色 + 建 QImage 约 450 ms（20 000 × 1056），首次切过去时
        这个代价会直接变成一次卡顿。与其让用户承担，不如分析完成后在事件
        循环里逐张预涂掉。

        手法：把 (db, channel_plane, step_cache, u8, qimg) 整体借用过来算一遍、
        存进缓存，再原样还回去。只在事件循环里单独调用（同一时刻不会有别的
        绘制路径在跑）才安全。
        """
        if db is None:
            return False
        if rows_per_semitone is not None and int(rows_per_semitone) != self.rows_per_semitone:
            self.rows_per_semitone = int(rows_per_semitone)

        saved = (self.db, self.channel_plane, self._step_cache, self.u8, self.qimg)
        try:
            self.db = db
            self.channel_plane = str(plane)
            self._step_cache = None
            self._regen_u8()
            return self._display_key() in self._u8_cache
        finally:
            (self.db, self.channel_plane, self._step_cache,
             self.u8, self.qimg) = saved

    def _remember_u8(self, key):
        try:
            self._u8_cache[key] = (self.u8, self.qimg)
        except (AttributeError, TypeError):
            return
        order = self._u8_cache_order
        if key in order:
            order.remove(key)
        order.append(key)
        # 从最旧的开始丢，直到回到上限以内
        while len(order) > self._U8_CACHE_KEEP:
            old = order.pop(0)
            self._u8_cache.pop(old, None)

    def _trim_u8_cache(self, keep_new=None):
        """只保留与当前显示参数同指纹的缓存，以及一份"刚刚换掉的那份"。

        音域、主题、显示域这类参数一改，旧指纹下的 QImage 就永远不会再被用到，
        留着纯占内存（每份可达几十 MB）。

        为什么还留 keep_new：取消设置的语义要求"回滚回去要原样"。若把旧指纹
        那份直接丢掉，用户改完主题再点取消，要么命不中缓存重算，要么更糟 ——
        命中了另一份不匹配的图。多留一份就同时满足"省内存"和"能回滚"。
        """
        cur = self._display_key()
        alive = {cur}
        if keep_new is not None and keep_new != cur:
            alive.add(keep_new)
        if all(k in alive for k in self._u8_cache):
            return
        self._u8_cache = {k: v for k, v in self._u8_cache.items() if k in alive}
        self._u8_cache_order = [k for k in self._u8_cache_order if k in alive]

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
