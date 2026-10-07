"""音频读写与结果导出。"""

import io
import os
import wave
import tempfile

import numpy as np

from wtpro.common import (
    DB_FLOOR, DEFAULT_ROWS_PER_SEMITONE, DEFAULT_TARGET_FPS, DEFAULT_WINDOW, MIDI_MAX,
    MIDI_MIN, _to_float
)
from wtpro.qt import QMediaContent, QUrl

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




# =========================================================================
# 多变体播放：一次预生成、靠音量热切换
# -------------------------------------------------------------------------
# 背景：切声道如果走 setMedia 重新装载，每次要付 ~50 ms，首次还有一次
# ~1.2 s 的 DirectShow 图重建（实测）。而"切换声道"在听感上只该是换一路
# 声音，不该有任何等待。
#
# 做法：为每个变体各生成一条临时 WAV、各建一个 QMediaPlayer，**三个一起
# 播放**，靠音量决定谁出声（目标 1.0，其余 0.0）。切换就是改音量：
# 实测 0.01~0.06 ms，且播放中热切换也成立。
#
# 注意不能只用一个播放器做等价的事：QMediaPlayer 没有声道选择/混音接口，
# 无法"从同一条立体声里只取左声道"。
# =========================================================================


class AudioVariants:
    """一组变体播放器，同一时刻只有一个在播。

    变体名与声道平面名一致：stereo / l / r。
    每个变体的 WAV 文件在装载阶段一次生成，播放器也在那时建好并暖过，
    之后切声道只是"暂停旧的、seek 新的、播新的"，实测 0.1 ms 量级。
    """

    def __init__(self, player_factory, parent=None):
        self._factory = player_factory
        self._parent = parent
        self._entries = {}          # name -> {"path": str|None, "player": QMediaPlayer|None}
        self.active = None
        self._volume = 1.0
        self._pos_ms = 0
        self._playing = False

    # ---------------- 构造 / 销毁 ----------------
    @staticmethod
    def write_pcm_wav(path, samples, sr):
        """把样本写成 16 bit 双声道 WAV。返回文件字节数。"""
        with open(path, "wb") as f:
            f.write(samples_to_wav_bytes(samples, sr))
        return os.path.getsize(path)

    def add(self, name, path):
        self._entries[name] = {"path": path, "player": None}

    def names(self):
        return tuple(self._entries)

    def has(self, name):
        return name in self._entries

    def close(self):
        """停掉并释放全部播放器与临时文件。"""
        for ent in self._entries.values():
            self._stop_player(ent)
        self._entries = {}
        self.active = None
        self._playing = False
        self._pos_ms = 0

    @staticmethod
    def _stop_player(ent):
        pl = ent.get("player")
        if pl is not None:
            try:
                pl.stop()
                pl.setMedia(QMediaContent())
            except Exception:
                pass
            try:
                pl.deleteLater()
            except Exception:
                pass
            ent["player"] = None
        p = ent.get("path")
        if p:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
            ent["path"] = None

    # ---------------- 播放器准备 ----------------
    def _player(self, name):
        ent = self._entries.get(name)
        if ent is None:
            return None
        pl = ent.get("player")
        if pl is None:
            pl = self._factory(self._parent)
            if pl is None:
                return None
            try:
                pl.setMedia(QMediaContent(QUrl.fromLocalFile(ent["path"])))
            except Exception as e:
                print(f"[media] 变体 {name} 装载失败: {e}")
                return None
            # 初始一律静音，出声与否完全由 _apply_volumes 决定
            try:
                pl.setVolume(0)
            except Exception:
                pass
            ent["player"] = pl
        return pl

    def _all(self):
        for name in self._entries:
            pl = self._entries[name].get("player")
            if pl is not None:
                yield pl

    def _apply_volumes(self):
        """目标变体出声、其余静音。这是"切换"的全部动作。"""
        for name, ent in self._entries.items():
            pl = ent.get("player")
            if pl is None:
                continue
            want = self._volume if name == self.active else 0.0
            try:
                pl.setVolume(int(round(max(0.0, min(1.0, want)) * 100)))
            except TypeError:
                pl.setVolume(max(0.0, min(1.0, want)))
            except Exception:
                pass

    # ---------------- 切换 ----------------
    def activate(self, name):
        """切换出声的变体。返回真正生效的变体名。

        切换的全部动作就是：把当前那一路暂停，把目标 seek 到它的位置再播。
        实测整段 0.1 ms 量级。

        **为什么静音的变体不一起播**：一开始的实现是三个一起播、只用音量选
        听谁，但实测两个问题：
          1. 播放中的播放器 seek 不准 —— setPosition 之后要 ~50 ms 才落到
             目标位置，这期间读回的还是旧值；
          2. 于是"切换前 seek 目标"和"定时把备胎对齐到出声那一路"两件事
             互相打架，越对越偏（实测能偏到 1.7 s）。
        改成备胎一律暂停之后，seek 落在暂停的播放器上是准的（实测 10 ms 内
        到位、停在哪就是哪），也就不需要再维护时间轴一致性了。
        """
        if name not in self._entries:
            return self.active
        if name == self.active:
            return self.active

        target = self._player(name)
        if target is None:
            return self.active

        # 先取出当前位置（此刻还在播，值是对的），再暂停当前那一路
        pos = self.position() if self._playing else self._pos_ms
        cur = self._entries.get(self.active, {}).get("player")
        if cur is not None and cur is not target:
            try:
                cur.setVolume(0)
                cur.pause()
            except Exception:
                pass

        try:
            target.setPosition(int(pos))
            if self._playing:
                target.setVolume(int(round(self._volume * 100)))
                target.play()
            else:
                target.setVolume(0)
        except Exception as e:
            print(f"[media] 切换到 {name} 失败: {e}")

        self.active = name
        self._pos_ms = int(pos)
        self._apply_volumes()
        return self.active

    def prepare(self):
        """把所有变体的播放器提前建好并"暖"一遍。

        必须做：某个变体第一次出声时要新建 QMediaPlayer 并让 DirectShow 把
        渲染图跑起来，实测那一下要 ~450 ms（同"首次 setVolume(100) 要 624 ms"
        是同一个冷启动）。放在装载阶段一次付掉，用户第一次切声道就不会卡。
        """
        for name in self._entries:
            pl = self._player(name)
            if pl is None:
                continue
            try:
                pl.setVolume(0)
                pl.play()
                pl.pause()
            except Exception:
                pass
        self._apply_volumes()

    # ---------------- 传输控制 ----------------
    def position(self):
        """当前出声变体的位置（毫秒）。只有它在播，所以它就是权威时间轴。"""
        if self.active:
            pl = self._entries.get(self.active, {}).get("player")
            if pl is not None:
                try:
                    return int(pl.position())
                except Exception:
                    pass
        return int(self._pos_ms)

    def set_position(self, ms):
        """跳转。只动当前出声的那一路 —— 备胎是暂停的，切过去时再 seek。"""
        self._pos_ms = int(max(0, ms))
        if self.active:
            pl = self._entries.get(self.active, {}).get("player")
            if pl is not None:
                try:
                    pl.setPosition(self._pos_ms)
                except Exception:
                    pass
                return
        for pl in self._all():
            try:
                pl.setPosition(self._pos_ms)
            except Exception:
                pass

    def play(self):
        """播放当前出声的那一路。

        只播一个：备胎保持暂停（切过去时再 seek）。见 activate 的说明。
        """
        if not self._entries:
            return
        if self.active is None:
            self.active = next(iter(self._entries))
        # 确保播放器都已建好（首次播放时才会真正创建）
        for name in self._entries:
            self._player(name)
        if not self._playing:
            self._playing = True
        pl = self._entries.get(self.active, {}).get("player")
        if pl is not None:
            try:
                pl.setPosition(self._pos_ms)
                pl.play()
            except Exception as e:
                print(f"[media] play 失败: {e}")
        self._apply_volumes()

    def pause(self):
        if self._playing and self.active:
            pl = self._entries.get(self.active, {}).get("player")
            if pl is not None:
                try:
                    self._pos_ms = int(pl.position())
                except Exception:
                    pass
        self._playing = False
        for pl in self._all():
            try:
                pl.pause()
            except Exception:
                pass

    def stop(self):
        self._playing = False
        self._pos_ms = 0
        for pl in self._all():
            try:
                pl.stop()
                pl.setPosition(0)
            except Exception:
                pass

    def set_volume(self, value_0_1):
        self._volume = max(0.0, min(1.0, float(value_0_1)))
        self._apply_volumes()


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
# 窗函数
# =========================================================================


def export_payload(mag, db, hop, sr, params=None, source_path=None,
                   hide_low=None, colormap=None, u8=None):
    """组装一份自描述的导出数据。

    mag : (n_frames, n_rows) float32，线性幅度，行 0 = MIDI_MAX
    db  : 同一形状的 dB（相对峰值，上限 0），便于直接画图
    """
    p = params or {}
    n_frames, n_rows = mag.shape
    midi_min = MIDI_MIN
    midi_max = MIDI_MAX
    rps = int(p.get("rows_per_semitone", DEFAULT_ROWS_PER_SEMITONE))
    frames = np.arange(n_frames, dtype=np.float64)
    times = frames * float(hop) / float(sr)
    rows = np.arange(n_rows)
    midis = (midi_max + 0.5) - (rows + 0.5) / float(rps)
    freqs = 440.0 * np.power(2.0, (midis - 69.0) / 12.0)
    return {
        "mag": np.asarray(mag, dtype=np.float32),
        "db": np.asarray(db, dtype=np.float32),
        "u8": (None if u8 is None else np.asarray(u8, dtype=np.uint8)),
        "time_s": times,
        "midi": midis,
        "freq_hz": freqs,
        # ---- 元数据（npz 里存成 0 维数组，csv 里写注释头）----
        "sr": int(sr),
        "hop": int(hop),
        "fps": float(sr) / float(hop),
        "midi_min": int(midi_min),
        "midi_max": int(midi_max),
        "rows_per_semitone": int(rps),
        "window_name": str(p.get("window_name", DEFAULT_WINDOW)),
        "target_fps": int(p.get("target_fps", DEFAULT_TARGET_FPS)),
        # 导出的是**当前显示的那一张平面**，所以必须记下它是哪一张 ——
        # 否则导出的 L 与 R 除了数值之外没有任何区别，事后对不上账。
        "channel_plane": str(p.get("channel_plane", "stereo")),
        "channel_mode": str(p.get("channel_mode", "stereo_only")),
        "db_floor": float(DB_FLOOR),
        "hide_low": ("" if hide_low is None else float(hide_low)),
        "colormap": ("" if colormap is None else str(colormap)),
        "duration_s": float(n_frames * hop) / float(sr),
        "source": ("" if source_path is None else os.path.basename(str(source_path))),
        "rows_are": "midi decreasing with row index; row 0 = MIDI_MAX",
    }




def export_result(path, payload, fmt=None):
    """把 payload 写到 path。fmt 为空时按扩展名推断。返回实际写入的路径。"""
    path = str(path)
    if fmt is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        fmt = {"npz": "npz", "npy": "npy", "csv": "csv",
               "bin": "raw(f32)", "raw": "raw(f32)", "f32": "raw(f32)"}.get(ext, "npz")
    mag = payload["mag"]
    db = payload["db"]
    meta = {k: v for k, v in payload.items()
            if k not in ("mag", "db", "u8") and not isinstance(v, np.ndarray)}

    if fmt == "npz":
        arrays = {"mag": mag, "db": db,
                  "time_s": payload["time_s"], "midi": payload["midi"],
                  "freq_hz": payload["freq_hz"]}
        if payload.get("u8") is not None:
            arrays["u8"] = payload["u8"]
        np.savez_compressed(path, **arrays, **meta)
        return path

    if fmt == "npy":
        np.save(path, mag)
        return path

    if fmt == "raw(f32)":
        mag.astype("<f4").tofile(path)
        return path

    if fmt == "csv":
        # 头部注释带元数据，之后第一列是时间，其余列按中音号命名
        lines = ["# wavetonepro export",
                 f"# shape(n_frames,n_rows)={db.shape[0]},{db.shape[1]}",
                 f"# rows_are={payload['rows_are']}"]
        for k in sorted(meta):
            lines.append(f"# {k}={meta[k]}")
        hdr = ["time_s"] + [f"midi{m:.3f}" for m in payload["midi"]]
        lines.append(",".join(hdr))
        t = payload["time_s"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(lines) + "\n")
            block = max(1, 200000 // max(1, db.shape[1]))
            for i in range(0, db.shape[0], block):
                sub = db[i:i + block]
                for j in range(sub.shape[0]):
                    f.write(f"{t[i + j]:.6f}," +
                            ",".join(f"{v:.3f}" for v in sub[j]) + "\n")
        return path

    raise ValueError(f"未知导出格式: {fmt}")



# =========================================================================
# 分析工作线程
# =========================================================================


EXPORT_FORMATS = ("npz", "npy", "csv", "raw(f32)")


