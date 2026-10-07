"""PyQt5 集中导入。需要 Qt 的模块都从这里取名字，
避免每个模块各写一份长 import 列表。"""

from PyQt5.QtCore import (
    Qt, QObject, QRectF, QPointF, QSize, pyqtSignal,
    QTimer, QUrl, QThread,
)
from PyQt5.QtGui import (QImage, QPainter, QColor, QPen, QFont,
                        QPixmap, QPolygonF)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QLabel,
    QComboBox,
    QSizePolicy,
    QMessageBox,
    QToolBar,
    QAction,
    QSlider,
    QStatusBar,
    QProgressBar,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QDialog,
    QFormLayout,
    QDialogButtonBox,
    QGroupBox,
    QCheckBox,
)

try:
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    HAS_MEDIA = True
except Exception as _e:
    QMediaPlayer = None
    QMediaContent = None
    HAS_MEDIA = False
    print(f"[media] QtMultimedia 不可用: {_e}")

class _BackendNotifier(QObject):
    changed = pyqtSignal(str)




_notifier = _BackendNotifier()


