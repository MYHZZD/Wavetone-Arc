"""后台分析线程。"""

import traceback

import numpy as np

from wtpro.common import (
    MIDI_MAX, MIDI_MIN, axis_samples, channel_mode_from_flags, mean_planes,
    mode_axes, stereo_average
)
from wtpro.qt import QThread, pyqtSignal
from wtpro.spectral import AnalysisCancelled, compute_db, compute_reassigned_spectrogram

# =========================================================================
# 取消异常定义在 spectral 里（分析层自己会抛），这里直接引用，
# 避免出现"worker 抛一个类、spectral 抛另一个类"导致取消失效。


class AnalysisWorker(QThread):
    """按声道模式跑 1~2 次分析，产出平面字典。

    done 回传 (planes_db, hop, planes_mag, ref)：

        planes_db / planes_mag : {"stereo"|"l"|"r": ndarray}
        hop                    : 帧步长
        ref                    : 各平面共用的 dB 归一参考（线性幅度）

    planes 的键顺序与 common.MODE_PLANES 一致（stereo → l → r），
    便于界面按下拉框顺序取用。
    """

    progress = pyqtSignal(int, str)
    done = pyqtSignal(object, int, object, float)
    failed = pyqtSignal(str)

    def __init__(self, samples, sr, params, parent=None):
        super().__init__(parent)
        self.samples = samples
        self.sr = sr
        self.params = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    # ------------------------------------------------------------------
    # 声道模式
    # ------------------------------------------------------------------
    def _channel_mode(self):
        """解析声道模式。

        优先读 get_params 已经算好的 channel_mode；没有就按 L / R 两个
        复选框现算；两者都缺则退回默认（stereo_only = 左右平均）。
        """
        mode = self.params.get("channel_mode")
        if mode:
            return mode
        return channel_mode_from_flags(self.params.get("channel_l", False),
                                       self.params.get("channel_r", False))

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self):
        try:
            planes_mag, hop, ref = self._analyze()
        except AnalysisCancelled:
            self.failed.emit("__cancelled__")
            return
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
            return

        if self._cancel:
            self.failed.emit("__cancelled__")
            return

        self.progress.emit(99, "计算 dB 中…")
        try:
            planes_db = {k: compute_db(v, ref=ref) for k, v in planes_mag.items()}
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
            return

        self.progress.emit(100, "完成")
        # 同时带回未归一化的线性幅度，供导出使用（db 只用于显示）
        self.done.emit(planes_db, hop, planes_mag, ref)

    def _analyze(self):
        """跑分析，返回 (planes_mag, hop, ref)。

        进度条按"本次要跑的分析趟数"分段：单声道模式走满 0~100，
        双声道模式第一趟 0~50、第二趟 50~100。这样两种模式下进度条的
        推进节奏一致，不会出现双声道模式进度条跑两遍的观感。
        """
        mode = self._channel_mode()
        axes = mode_axes(mode)

        planes_mag = {}
        hop = None

        if not axes:
            # ---- stereo_only：只跑一次，喂左右平均 ----
            mag, hop = self._run_once(stereo_average(self.samples),
                                      self._scaled_progress(0, 1), self._cancel_cb)
            planes_mag["stereo"] = mag
        else:
            # ---- l_only / r_only / lr_both：逐声道各跑一次 ----
            n = len(axes)
            for i, axis in enumerate(axes):
                if self._cancel:
                    raise AnalysisCancelled()
                mag, hop = self._run_once(
                    axis_samples(self.samples, axis),
                    self._scaled_progress(i, n),
                    self._cancel_cb,
                )
                planes_mag["l" if axis == 0 else "r"] = mag

        if not planes_mag:
            raise RuntimeError("没有产出任何声道平面")

        # stereo 平面：L、R 都在手时按**幅度**平均得出，不必再分析一遍。
        # （先把两路样本加起来再分析是代数平均，反相内容会互相抵消。）
        # 只有单侧时不合成 —— 拿单侧冒充立体声会误导。
        if "l" in planes_mag and "r" in planes_mag:
            planes_mag = {
                "stereo": mean_planes(planes_mag["l"], planes_mag["r"]),
                "l": planes_mag["l"],
                "r": planes_mag["r"],
            }

        # 各平面共用一个归一参考：各自除以自己的峰值会把左右电平差抹平，
        # 三张图之间也没法比较。
        ref = max(float(v.max()) for v in planes_mag.values() if v.size)
        return planes_mag, hop, ref

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _cancel_cb(self):
        return self._cancel

    def _scaled_progress(self, i, n):
        """把第 i / n 趟的 0~1 映射到整体进度。"""
        def cb(frac):
            pct = int(max(0.0, min(1.0, (i + float(frac)) / float(n))) * 100)
            self.progress.emit(pct, f"分析中… {pct}%")
        return cb

    def _run_once(self, samples_1d, progress_cb, cancel_cb):
        return compute_reassigned_spectrogram(
            samples_1d,
            self.sr,
            midi_min=MIDI_MIN,
            midi_max=MIDI_MAX,
            rows_per_semitone=self.params["rows_per_semitone"],
            target_fps=self.params["target_fps"],
            window_name=self.params["window_name"],
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
            # 插值总开关（频率 + 时间两个方向），缺省开启
            interp=self.params.get("interp", True),
        )


# =========================================================================
# 参数对话框
# =========================================================================
