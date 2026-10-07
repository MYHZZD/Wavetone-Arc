"""WavetonePro 入口。

实现已经按功能拆进同目录的 wtpro 包：

    wtpro/common.py       公共常量与坐标换算（音高/MIDI/频率、和弦、默认参数）
    wtpro/qt.py           PyQt5 集中导入
    wtpro/backends.py     计算后端（CPU/CuPy）与 MIDI 的输出探测、切换、状态查询
    wtpro/color.py        配色：内置表、WaveTone 渐变、线性域饱和起点换算
    wtpro/audio.py        音频读写与结果导出
    wtpro/spectral.py     频谱分析：窗函数、多分辨率分组、STFT、相位重分配
    wtpro/render.py       显示与后处理：dB 换算、显示域映射、阶梯滤镜、曲线、上色
    wtpro/worker.py       后台分析线程
    wtpro/dialogs.py      各类设置对话框
    wtpro/spectrogram.py  频谱图视图与钢琴卷帘
    wtpro/app.py          主窗口与 main()

本文件保留项目根位置和同名脚本，因为 PyInstaller 的 ENTRY_SCRIPT 指向它
（见 build/wavetonepro.spec），打包流程不需要任何修改。

直接运行：

    python wavetonepro.py [音频文件]
"""
import os
import sys

# 以脚本方式运行时，确保能 import 同目录的 wtpro 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wtpro.app import MainWindow, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
