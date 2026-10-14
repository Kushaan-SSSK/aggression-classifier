# Kushaan Sharma
"""Desktop app for scoring resident-intruder videos with the trained classifiers.

Videos   pick a folder of recordings, mark the cage and the intruder entry once per video, press Run
Results  per-video latency / bouts / time spent, an ethogram of every video, group comparisons
Review   the processed video next to the model's probability trace; jump from bout to bout

Launch with `python -m behavior_pipeline gui` (or Aggression Scorer.bat). Every stage runs as the same command-line
call the README lists, in its own process, so closing a window never leaves a half-written file behind.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QObject, QPointF, QRectF, QSettings, QSortFilterProxyModel, Qt,
                            QTimer, QUrl, Signal)
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog,
                               QFrame, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QHeaderView, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QSlider, QStackedWidget, QTableView, QVBoxLayout, QWidget)

import pyqtgraph as pg  # noqa: E402  (after PySide6, so pyqtgraph binds to it)

from . import style
from .style import C, GROUP_COLOURS, ElidedLabel, button, label, line, style_plot
from .workspace import (REPO, TEMPLATE_CONFIG, VideoSetup, Workspace, available_models, batch_log_settings, probe,
                        scan_folder, unique_names)

METRICS = {"latency_s": "Latency to first attack (s)", "episodes": "Episodes", "bouts": "Bouts",
           "total_s": "Time attacking (s)", "pct_time": "Time attacking (%)", "mean_bout_s": "Mean bout length (s)",
           "longest_bout_s": "Longest bout (s)"}
MARGIN_X, MARGIN_Y = 40, 28           # 4 px grid


def fmt_time(s: float | None) -> str:
    """0:27.4 style; – when missing."""
    if s is None or not np.isfinite(s):
        return "–"
    m, sec = divmod(max(float(s), 0.0), 60)
    return f"{int(m)}:{sec:04.1f}"


def fmt_s(s: float | None) -> str:
    return "–" if s is None or not np.isfinite(s) else f"{s:.1f} s"


def page(spacing: int = 12) -> tuple[QWidget, QVBoxLayout]:
    """A white page with the standard margins; returns the widget and its layout."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(MARGIN_X, MARGIN_Y, MARGIN_X, MARGIN_Y)
    lay.setSpacing(spacing)
    return w, lay


def row(*items, spacing: int = 8) -> QHBoxLayout:
    """Widgets, layouts, fixed gaps (int) and stretches (None) in a row."""
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    for x in items:
        if x is None:
            lay.addStretch(1)
        elif isinstance(x, int):
            lay.addSpacing(x)
        elif isinstance(x, (QHBoxLayout, QVBoxLayout)):
            lay.addLayout(x)
        else:
            lay.addWidget(x)
    return lay


def plain_table(view: QTableView) -> QTableView:
    view.setShowGrid(False)
    view.setAlternatingRowColors(True)
    view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    view.verticalHeader().setVisible(False)
    view.verticalHeader().setDefaultSectionSize(36)
    h = view.horizontalHeader()
    h.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    h.setHighlightSections(False)
    h.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    h.setStretchLastSection(True)
    h.setMinimumSectionSize(64)
    view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    return view


# ---------------------------------------------------------------------------------------------------------------
# table models
# ---------------------------------------------------------------------------------------------------------------
class DataFrameModel(QAbstractTableModel):
    def __init__(self, headers: dict[str, str] | None = None, formats: dict | None = None):
        super().__init__()
        self.df = pd.DataFrame()
        self.headers = headers or {}
        self.formats = formats or {}

    def set(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df.columns)

    def _numeric(self, col: str) -> bool:
        return pd.api.types.is_numeric_dtype(self.df[col])

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        col = self.df.columns[index.column()]
        v = self.df.iat[index.row(), index.column()]
        num = isinstance(v, (int, float, np.integer, np.floating))
        if role == Qt.ItemDataRole.DisplayRole:
            if v is None or (num and not np.isfinite(v)) or (isinstance(v, str) and not v):
                return "–"
            if col in self.formats:
                return self.formats[col](v)
            return f"{v:g}" if num else str(v)
        if role == Qt.ItemDataRole.UserRole:  # sort key
            return float(v) if num and v == v else (str(v) if v is not None else "")
        if role == Qt.ItemDataRole.TextAlignmentRole and num:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        c = str(self.df.columns[section])
        if role == Qt.ItemDataRole.DisplayRole:
            return self.headers.get(c, c)
        if role == Qt.ItemDataRole.TextAlignmentRole and self._numeric(c):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None


def sortable(model: QAbstractTableModel) -> QSortFilterProxyModel:
    proxy = QSortFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.setSortRole(Qt.ItemDataRole.UserRole)
    return proxy


STATUS = {"setup": "Needs setup", "ready": "Ready", "scored": "Scored", "changed": "Changed, will be redone", "progress": "Started"}


class VideoTableModel(QAbstractTableModel):
    COLS = ["", "Video", "Length", "Intruder entry", "Cage", "Group", "Status"]
    changed_by_user = Signal()

    def __init__(self):
        super().__init__()
        self.rows: list[VideoSetup] = []
        self.use: list[bool] = []
        self.status: list[str] = []

    def set(self, rows: list[VideoSetup], use: list[bool] | None = None) -> None:
        self.beginResetModel()
        self.rows = rows
        self.use = use or [True] * len(rows)
        self.status = ["setup"] * len(rows)
        self.endResetModel()

    def refresh(self) -> None:
        if self.rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, len(self.COLS) - 1))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.COLS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.COLS[section]
        return None

    def flags(self, index):
        f = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            f |= Qt.ItemFlag.ItemIsUserCheckable
        if index.column() in (3, 5):
            f |= Qt.ItemFlag.ItemIsEditable
        return f

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        s = self.rows[r]
        if role == Qt.ItemDataRole.CheckStateRole and c == 0:
            return Qt.CheckState.Checked if self.use[r] else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.DisplayRole:
            if c == 1:
                return s.name
            if c == 2:
                return fmt_time(s.duration_s)[:-2] if s.fps else "?"
            if c == 3:
                return "Not set" if s.entry_s is None else fmt_time(s.entry_s)
            if c == 4:
                return "Not set" if s.crop is None else "Set"
            if c == 5:
                return s.group
            if c == 6:
                return STATUS[self.status[r]]
        if role == Qt.ItemDataRole.EditRole:
            return {3: f"{s.entry_s:.1f}" if s.entry_s is not None else "", 5: s.group}.get(c)
        if role == Qt.ItemDataRole.ForegroundRole:
            if (c == 3 and s.entry_s is None) or (c == 4 and s.crop is None) or (c == 6 and self.status[r] in ("setup", "changed")):
                return QBrush(QColor(C["warn"]))
            if c in (2, 6):
                return QBrush(QColor(C["muted"]))
        if role == Qt.ItemDataRole.ToolTipRole:
            return s.source if c == 1 else ("Double-click to set up" if c in (3, 4) else None)
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        r, c = index.row(), index.column()
        if c == 0 and role == Qt.ItemDataRole.CheckStateRole:
            self.use[r] = Qt.CheckState(value) == Qt.CheckState.Checked
        elif c == 3 and role == Qt.ItemDataRole.EditRole:
            try:
                self.rows[r].entry_s = max(float(str(value).replace("s", "").strip()), 0.0)
            except ValueError:
                return False
        elif c == 5 and role == Qt.ItemDataRole.EditRole:
            self.rows[r].group = str(value).strip()
        else:
            return False
        self.dataChanged.emit(self.index(r, 0), self.index(r, len(self.COLS) - 1))
        self.changed_by_user.emit()
        return True


# ---------------------------------------------------------------------------------------------------------------
# per-video setup: cage crop and intruder entry
# ---------------------------------------------------------------------------------------------------------------
class FrameView(QGraphicsView):
    """A video frame; drag to draw the crop rectangle."""
    crop_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.pix)
        pen = QPen(QColor(C["accent"]), 2)
        pen.setCosmetic(True)
        self.rect = QGraphicsRectItem()
        self.rect.setPen(pen)
        self.rect.setBrush(QBrush(QColor(15, 108, 189, 30)))
        self.rect.setZValue(1)
        self.rect.hide()
        self.scene().addItem(self.rect)
        self._origin: QPointF | None = None
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor(C["track"]))
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def show_frame(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        first = self.pix.pixmap().isNull()
        self.pix.setPixmap(QPixmap.fromImage(img))
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        if first:
            self.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self.pix.pixmap().isNull():
            self.fitInView(self.pix, Qt.AspectRatioMode.KeepAspectRatio)

    def set_crop(self, crop) -> None:
        if crop is None:
            self.rect.hide()
        else:
            self.rect.setRect(QRectF(*crop))
            self.rect.show()

    def crop(self) -> tuple[int, int, int, int] | None:
        if not self.rect.isVisible():
            return None
        r = self.rect.rect().normalized().intersected(self.scene().sceneRect())
        x, y, w, h = int(r.x()), int(r.y()), int(r.width()) // 2 * 2, int(r.height()) // 2 * 2
        return (x, y, w, h) if w >= 20 and h >= 20 else None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._origin = self.mapToScene(e.position().toPoint())
            self.rect.setRect(QRectF(self._origin, self._origin))
            self.rect.show()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self.rect.setRect(QRectF(self._origin, self.mapToScene(e.position().toPoint())).normalized())

    def mouseReleaseEvent(self, e):
        if self._origin is not None:
            self._origin = None
            if self.crop() is None:
                self.rect.hide()
            self.crop_changed.emit()


class SetupDialog(QDialog):
    """Mark the cage and the intruder entry, one video after another."""

    def __init__(self, setups: list[VideoSetup], index: int, parent=None):
        super().__init__(parent)
        self.setups, self.i = setups, index
        self.cap = None
        self.entry: float | None = None
        self.setWindowTitle("Set up videos")
        self.resize(1040, 820)

        self.title = label("", "subtitle")
        self.counter = label("", "muted")
        howto = label("Drag a box around the cage floor. Then move to the frame where the intruder is put in and click "
                      "Mark entry. Arrow keys step 0.1 s, Shift + arrow 1 s.", "muted", wrap=True)
        self.view = FrameView()
        self.view.crop_changed.connect(self._update_state)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(self._seek)
        self.time = label("")
        self.time.setMinimumWidth(100)
        self.time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.mark = button("Mark entry", slot=self._mark)
        self.state = label("", "muted")
        whole = button("Use whole frame", slot=self._full_frame)
        self.same = QCheckBox("Use this box for other videos of the same size")
        self.same.setChecked(True)
        self.back = button("Back", slot=lambda: self._move(-1))
        self.next = button("Next", "primary", lambda: self._move(1))
        self.next.setDefault(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)
        lay.addLayout(row(self.title, None, self.counter))
        lay.addWidget(howto)
        lay.addWidget(self.view, 1)
        lay.addLayout(row(self.slider, self.time, spacing=12))
        lay.addLayout(row(self.mark, whole, 12, self.state, None))
        lay.addSpacing(8)
        lay.addLayout(row(self.same, None, self.back, self.next))
        self._load()

    def keyPressEvent(self, e):
        step = 10 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        if e.key() == Qt.Key.Key_Left:
            self.slider.setValue(self.slider.value() - step)
        elif e.key() == Qt.Key.Key_Right:
            self.slider.setValue(self.slider.value() + step)
        elif e.key() == Qt.Key.Key_E:
            self._mark()
        else:
            super().keyPressEvent(e)

    def _load(self) -> None:
        import cv2

        s = self.setups[self.i]
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(s.source)
        self.title.setText(s.name)
        self.counter.setText(f"Video {self.i + 1} of {len(self.setups)}")
        self.entry = s.entry_s
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(int(s.duration_s * 10) - 1, 0))
        self.slider.setValue(int(round((s.entry_s or 0.0) * 10)))
        self.slider.blockSignals(False)
        self.view.pix.setPixmap(QPixmap())
        self._seek(self.slider.value())
        self.view.set_crop(s.crop)
        self.back.setEnabled(self.i > 0)
        self.next.setText("Done" if self.i == len(self.setups) - 1 else "Next")
        self._update_state()

    def _seek(self, v: int) -> None:
        import cv2

        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_MSEC, v * 100.0)
        ok, frame = self.cap.read()
        if ok:
            self.view.show_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.time.setText(f"{fmt_time(v / 10)} / {fmt_time(self.setups[self.i].duration_s)}")

    def _mark(self) -> None:
        self.entry = self.slider.value() / 10.0
        self._update_state()

    def _update_state(self) -> None:
        box = "Box set" if self.view.crop() is not None else "No box yet"
        entry = f"entry at {fmt_time(self.entry)}" if self.entry is not None else "entry not marked"
        self.state.setText(f"{box}, {entry}")

    def _full_frame(self) -> None:
        s = self.setups[self.i]
        self.view.set_crop((0, 0, s.width // 2 * 2, s.height // 2 * 2))
        self._update_state()

    def _store(self) -> None:
        s = self.setups[self.i]
        s.crop = self.view.crop()
        s.entry_s = self.entry
        if s.crop and self.same.isChecked():
            for o in self.setups:
                if o.crop is None and (o.width, o.height) == (s.width, s.height):
                    o.crop = s.crop

    def _move(self, step: int) -> None:
        self._store()
        j = self.i + step
        if 0 <= j < len(self.setups):
            self.i = j
            self._load()
        elif step > 0:
            self.accept()

    def done(self, r):
        if r == QDialog.DialogCode.Rejected:
            self._store()  # closing the window keeps what was marked
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().done(r)


# ---------------------------------------------------------------------------------------------------------------
# running the stages
# ---------------------------------------------------------------------------------------------------------------
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Runner(QObject):
    """Runs the stages one after another, each as its own process.

    Each stage is started with python.exe and CREATE_NO_WINDOW: it gets a hidden console, and every console program it
    starts in turn (ffmpeg, DeepLabCut's workers) shares that console instead of opening a black window of its own.
    """
    stage_started = Signal(int)
    stage_finished = Signal(int, bool, float)
    line = Signal(str)
    finished = Signal(bool)
    _ended = Signal(int)

    def __init__(self):
        super().__init__()
        self.proc: subprocess.Popen | None = None
        self.stages = []
        self.i = -1
        self.t0 = 0.0
        self.stopped = False
        self._ended.connect(self._done)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, stages) -> None:
        self.stages, self.i, self.stopped = stages, -1, False
        self._next()

    def _next(self) -> None:
        self.i += 1
        if self.i >= len(self.stages):
            self.finished.emit(True)
            return
        st = self.stages[self.i]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        cmd = [st.python, "-u", "-m", "behavior_pipeline", *st.args]
        self.t0 = time.time()
        self.stage_started.emit(self.i)
        self.line.emit(f"$ {Path(st.python).name} -m behavior_pipeline {' '.join(st.args)}")
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        except OSError as exc:
            self.line.emit(f"could not start {st.python}: {exc}")
            self.stage_finished.emit(self.i, False, 0.0)
            self.finished.emit(False)
            return
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen) -> None:
        """Reader thread: forward output line by line (progress bars write \\r), then report the exit code."""
        buf = b""
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
            if not chunk:
                break
            buf += chunk.replace(b"\r", b"\n")
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if ln.strip():
                    self.line.emit(ln.decode("utf-8", "replace").rstrip())
        if buf.strip():
            self.line.emit(buf.decode("utf-8", "replace").rstrip())
        self._ended.emit(proc.wait())

    def _done(self, code: int) -> None:
        ok = code == 0 and not self.stopped
        self.stage_finished.emit(self.i, ok, time.time() - self.t0)
        if ok:
            self._next()
        else:
            self.finished.emit(False)

    def stop(self) -> None:
        self.stopped = True
        if self.running():
            # kill the whole tree: ffmpeg and DeepLabCut start their own children
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True, creationflags=NO_WINDOW)
            else:
                self.proc.kill()


# ---------------------------------------------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------------------------------------------
class VideosPage(QWidget):
    results_ready = Signal()
    workspace_changed = Signal(object)

    def __init__(self, settings: QSettings, template: Path):
        super().__init__()
        self.settings, self.template = settings, template
        self.ws: Workspace | None = None
        self.folder: Path | None = None
        self.runner = Runner()
        self.runner.stage_started.connect(self._stage_started)
        self.runner.stage_finished.connect(self._stage_finished)
        self.runner.line.connect(self._line)
        self.runner.finished.connect(self._finished)
        self.stages = []
        self.n_videos = 0
        self.done_in_stage: set[str] = set()
        self.models = available_models(template)

        # first screen: nothing chosen yet
        empty, el = page(spacing=8)
        el.addWidget(label("Score a folder of videos", "title"))
        el.addWidget(label("Choose the folder with your resident-intruder recordings. You mark the cage and the moment the "
                           "intruder is put in once per video; everything else runs on its own.", "muted", wrap=True))
        el.addSpacing(16)
        el.addLayout(row(button("Choose folder…", "primary", self.choose_folder), None))
        if not self.models:
            el.addSpacing(8)
            el.addWidget(label(f"No trained model found next to {template}. See USER_GUIDE.md, step 1.4.", "muted", wrap=True))
        el.addStretch(1)
        for i in range(el.count()):
            w = el.itemAt(i).widget()
            if w is not None:
                w.setMaximumWidth(640)

        # folder chosen
        self.folder_name = label("", "title")
        self.results_path = ElidedLabel("", "caption")
        self.model_checks: dict[str, QCheckBox] = {}
        models_row = QHBoxLayout()
        models_row.setSpacing(20)
        if len(self.models) > 1:
            models_row.addWidget(label("Score:", "muted"))
            for clf in self.models:
                cb = QCheckBox(clf.replace("_", " "))
                cb.setChecked(clf == "Attack")
                cb.setToolTip(f"Model trained {self.models[clf]['trained']}")
                self.model_checks[clf] = cb
                models_row.addWidget(cb)
        elif self.models:
            clf = next(iter(self.models))
            self.model_checks[clf] = QCheckBox()
            self.model_checks[clf].setChecked(True)
        models_row.addStretch(1)

        self.table_model = VideoTableModel()
        self.table_model.changed_by_user.connect(self._save_setups)
        self.table = plain_table(QTableView())
        self.table.setModel(self.table_model)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)  # double-click handled in _double_clicked
        self.table.doubleClicked.connect(self._double_clicked)
        self.summary = label("", "muted")

        self.run_btn = button("Run analysis", "primary", self.run)
        self.stop_btn = button("Stop", slot=self.stop)
        self.stop_btn.hide()
        self.log_btn = button("Show log", slot=self._toggle_log)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.status = label("", "muted")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setMinimumHeight(160)
        self.log.hide()

        folder_view, fl = page(spacing=8)
        fl.addLayout(row(self.folder_name, None, button("Change folder…", slot=self.choose_folder),
                         button("Save results elsewhere…", slot=self.choose_output,
                                tip="Results normally go into an 'aggression_analysis' folder next to the videos")))
        fl.addWidget(self.results_path)
        fl.addSpacing(16)
        if len(self.models) > 1:
            fl.addLayout(models_row)
            fl.addSpacing(8)
        fl.addWidget(self.table, 1)
        fl.addSpacing(8)
        fl.addLayout(row(button("Set up videos…", slot=lambda: self.setup_videos(self._first_selected_or_unready())),
                         button("Copy setup to all", slot=self.copy_to_all,
                                tip="Give every video the box and entry time of the selected one"),
                         button("Import SimBA batch log…", slot=self.load_batch_log,
                                tip="Take boxes and entry times from SimBA's batch_process_log.json"),
                         None, self.summary))
        fl.addSpacing(24)
        fl.addWidget(self.progress)
        fl.addLayout(row(self.status, None, self.log_btn, self.stop_btn, self.run_btn))
        fl.addWidget(self.log)

        self.stack = QStackedWidget()
        self.stack.addWidget(empty)
        self.stack.addWidget(folder_view)
        QVBoxLayout(self).setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self.stack)

    # ---- folder and setups --------------------------------------------------------------------------------------
    def choose_folder(self) -> None:
        start = self.settings.value("last_folder", str(Path.home()))
        d = QFileDialog.getExistingDirectory(self, "Folder with the recordings", start)
        if d:
            self.open_folder(Path(d))

    def choose_output(self) -> None:
        if self.folder is None:
            return
        d = QFileDialog.getExistingDirectory(self, "Where to save the analysis (a short path such as C:\\Analysis is safest)",
                                             str(self.ws.root.parent if self.ws else self.folder))
        if not d:
            return
        root = Path(d) / f"aggression_{self.folder.name}"[:60]
        self.settings.setValue(f"output/{self.folder.resolve()}", str(root))
        self.open_folder(self.folder)

    def open_folder(self, folder: Path) -> None:
        paths = scan_folder(folder)
        if not paths:
            QMessageBox.information(self, "No videos", f"There are no video files (mp4, avi, mov, mkv) in\n{folder}")
            return
        self.settings.setValue("last_folder", str(folder))
        self.folder = folder
        custom = self.settings.value(f"output/{folder.resolve()}", "")
        self.ws = Workspace(Path(custom)) if custom else Workspace.for_folder(folder)
        self.folder_name.setText(folder.name)
        self.results_path.setText(f"Results are saved in {self.ws.root}")
        saved = self.ws.load_setups()
        from_log = batch_log_settings(folder)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            rows, use = [], []
            for p, name in zip(paths, unique_names(paths)):
                key = str(p.resolve())
                s = saved.get(key) or VideoSetup(source=str(p), name=name)
                s.source = str(p)
                info = probe(p)
                s.fps, s.frames, s.width, s.height = info["fps"], info["frames"], info["width"], info["height"]
                if (s.entry_s is None or s.crop is None) and p.stem in from_log:
                    entry, crop = from_log[p.stem]
                    s.entry_s = s.entry_s if s.entry_s is not None else entry
                    s.crop = s.crop or crop
                rows.append(s)
                use.append(key not in saved or getattr(saved[key], "note", "") != "unused")
        finally:
            QApplication.restoreOverrideCursor()
        self.table_model.set(rows, use)
        self.stack.setCurrentIndex(1)
        self._update_status()
        self.workspace_changed.emit(self.ws)

    def _save_setups(self) -> None:
        if self.ws is None:
            return
        for s, u in zip(self.table_model.rows, self.table_model.use):
            s.note = "" if u else "unused"
        self.ws.save_setups(self.table_model.rows)
        self._update_status()

    def _update_status(self) -> None:
        rows = self.table_model.rows
        if self.ws is not None:
            traces = self.ws.results_dir / "traces"
            for i, s in enumerate(rows):
                if not s.ready:
                    st = "setup"
                elif self.ws.changed(s):
                    st = "changed"
                elif (traces / f"{s.name}.csv").exists():
                    st = "scored"
                elif self.ws.processed(s.name):
                    st = "progress"
                else:
                    st = "ready"
                self.table_model.status[i] = st
        self.table_model.refresh()
        used = [s for s, u in zip(rows, self.table_model.use) if u]
        need = sum(not s.ready for s in used)
        self.summary.setText(f"{len(used)} of {len(rows)} videos selected")
        if not self.runner.running():
            scored = used and all(self.table_model.status[i] == "scored" for i, u in enumerate(self.table_model.use) if u)
            self.status.setText(f"{need} video{'s' if need != 1 else ''} still need setting up. Double-click a video to set it up."
                                if need else "All selected videos are scored." if scored else "Ready to run.")

    def _double_clicked(self, idx) -> None:
        if idx.column() == 5:
            self.table.edit(idx)
        elif idx.column() != 0:
            self.setup_videos(idx.row())

    def _first_selected_or_unready(self) -> int:
        sel = self.table.selectionModel().selectedRows()
        if sel:
            return sel[0].row()
        for i, s in enumerate(self.table_model.rows):
            if not s.ready and self.table_model.use[i]:
                return i
        return 0

    def setup_videos(self, row_: int) -> None:
        if self.table_model.rows:
            SetupDialog(self.table_model.rows, row_, self).exec()
            self._save_setups()

    def copy_to_all(self) -> None:
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            QMessageBox.information(self, "Copy setup", "First click the video whose box and entry time should be copied.")
            return
        src = self.table_model.rows[sel[0].row()]
        if not src.ready:
            QMessageBox.information(self, "Copy setup", f"{src.name} is not set up yet.")
            return
        others = [s for s in self.table_model.rows if s is not src]
        if QMessageBox.question(self, "Copy setup", f"Give the other {len(others)} videos the box and entry time "
                                f"({fmt_time(src.entry_s)}) of {src.name}?") != QMessageBox.StandardButton.Yes:
            return
        for s in others:
            s.crop, s.entry_s = src.crop, src.entry_s
        self._save_setups()

    def load_batch_log(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "SimBA batch_process_log.json", str(self.folder or Path.home()), "JSON (*.json)")
        if not f:
            return
        from ..video.adopt import _hms_to_s, load_batch_log

        n = 0
        for stem, e in load_batch_log(Path(f)).items():
            for s in self.table_model.rows:
                if Path(s.source).stem == stem:
                    c = e.get("crop_settings") or {}
                    if e.get("crop") and c:
                        s.crop = (int(c["top_left_x"]), int(c["top_left_y"]), int(c["width"]), int(c["height"]))
                    s.entry_s = _hms_to_s((e.get("clip_settings") or {}).get("start", "0")) if e.get("clip") else 0.0
                    n += 1
        self._save_setups()
        QMessageBox.information(self, "Batch log", f"Box and entry time taken for {n} video(s).")

    # ---- running ------------------------------------------------------------------------------------------------
    def _toggle_log(self) -> None:
        self.log.setVisible(not self.log.isVisible())
        self.log_btn.setText("Hide log" if self.log.isVisible() else "Show log")

    def run(self) -> None:
        if self.ws is None:
            return
        clfs = [c for c, cb in self.model_checks.items() if cb.isChecked()]
        if not clfs:
            QMessageBox.information(self, "Run", "Tick at least one behaviour to score.")
            return
        used = [s for s, u in zip(self.table_model.rows, self.table_model.use) if u]
        if not used:
            QMessageBox.information(self, "Run", "Tick at least one video.")
            return
        unready = [s for s in used if not s.ready]
        if unready:
            if QMessageBox.question(self, "Some videos need setting up",
                                    f"{len(unready)} video(s) still need a cage box or an intruder entry. Set them up now?") \
                    == QMessageBox.StandardButton.Yes:
                self.setup_videos(self.table_model.rows.index(unready[0]))
            return
        deepest = self.ws.longest_path([s.name for s in used])
        if deepest > 255:
            QMessageBox.warning(self, "Folder path too long",
                                f"Some files would get paths {deepest} characters long, and Windows stops at 260.\n\n"
                                "Click 'Save results elsewhere' and pick a short folder (for example C:\\Analysis).")
            return
        self.log.clear()
        for s in [s for s in used if self.ws.changed(s)]:
            self.ws.forget(s.name)
            self._line(f"{s.name}: box or entry time changed, earlier outputs removed")
        self.ws.create(clfs, self.folder)
        self.ws.write_manifest(used)
        self.ws.save_setups(self.table_model.rows)
        self.n_videos = len(used)
        self.stages = self.ws.stages(clfs)
        self.progress.setRange(0, len(self.stages) * max(self.n_videos, 1))
        self.progress.setValue(0)
        self.progress.show()
        self.run_btn.hide()
        self.stop_btn.show()
        self.runner.start(self.stages)

    def stop(self) -> None:
        if QMessageBox.question(self, "Stop", "Stop the analysis? Finished videos are kept; the rest is redone next time.") \
                == QMessageBox.StandardButton.Yes:
            self.runner.stop()

    def _say(self, i: int, extra: str = "") -> None:
        self.status.setText(f"Step {i + 1} of {len(self.stages)}: {self.stages[i].title}{extra}")

    def _stage_started(self, i: int) -> None:
        self.done_in_stage = set()
        self._say(i)
        self.progress.setValue(i * max(self.n_videos, 1))

    _VIDEO_DONE = re.compile(r"INFO (\S+): (?:pose )?done\b")

    def _line(self, text: str) -> None:
        self.log.appendPlainText(text)
        i = self.runner.i
        m = self._VIDEO_DONE.search(text)
        if 0 <= i < len(self.stages) and m:
            self.done_in_stage.add(m.group(1))
            k = min(len(self.done_in_stage), self.n_videos)
            self._say(i, f" ({k} of {self.n_videos} videos)")
            self.progress.setValue(i * max(self.n_videos, 1) + k)

    def _stage_finished(self, i: int, ok: bool, secs: float) -> None:
        took = f"{secs / 60:.1f} min" if secs >= 60 else f"{secs:.0f} s"
        self.log.appendPlainText(f"{'Finished' if ok else 'FAILED'}: {self.stages[i].title} ({took})")

    def _finished(self, ok: bool) -> None:
        self.run_btn.show()
        self.stop_btn.hide()
        self.progress.hide()
        self._update_status()
        if ok:
            self.status.setText("Finished. The results are on the Results page.")
            self.results_ready.emit()
        elif self.runner.stopped:
            self.status.setText("Stopped.")
        else:
            self.status.setText(f"{self.stages[self.runner.i].title} failed. Press Run analysis to try again.")
            if not self.log.isVisible():
                self._toggle_log()
            tail = "\n".join(self.log.toPlainText().splitlines()[-12:])
            QMessageBox.warning(self, "Something went wrong", f"{self.stages[self.runner.i].title} failed.\n\nLast lines of the log:\n\n{tail}")


# ---------------------------------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------------------------------
class ResultsPage(QWidget):
    review_requested = Signal(str, float)

    def __init__(self):
        super().__init__()
        self.ws: Workspace | None = None
        self.summary = pd.DataFrame()
        self.bouts = pd.DataFrame()
        self.etho_rows: list[str] = []

        self.clf_box = QComboBox()
        self.clf_box.currentTextChanged.connect(self.redraw)
        self.headline = label("", wrap=True)
        self.ethogram = pg.PlotWidget()
        style_plot(self.ethogram, bottom="Seconds after intruder entry")
        self.ethogram.setMouseEnabled(x=True, y=False)
        self.ethogram.scene().sigMouseClicked.connect(self._ethogram_click)
        self.ethogram.setToolTip("Each block is one attack bout. Click one to watch it.")

        fmt1 = lambda v: f"{v:.1f}"
        self.table_model = DataFrameModel(
            headers={"video": "Video", "group": "Group", "latency_s": "Latency (s)", "episodes": "Episodes", "bouts": "Bouts",
                     "total_s": "Time attacking (s)", "pct_time": "Time attacking (%)", "mean_bout_s": "Mean bout (s)",
                     "longest_bout_s": "Longest bout (s)"},
            formats={"latency_s": fmt1, "total_s": fmt1, "pct_time": fmt1, "mean_bout_s": fmt1, "longest_bout_s": fmt1})
        self.proxy = sortable(self.table_model)
        self.table = plain_table(QTableView())
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self._table_review)
        self.table.setToolTip("Double-click a video to watch its first attack")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.metric_box = QComboBox()
        for k, v in METRICS.items():
            self.metric_box.addItem(v, k)
        self.metric_box.currentIndexChanged.connect(self.redraw)
        self.group_plot = pg.PlotWidget()
        style_plot(self.group_plot)
        self.group_plot.setMouseEnabled(x=False, y=False)
        self.group_box = QWidget()
        gl = QVBoxLayout(self.group_box)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(8)
        gl.addLayout(row(label("By group", "strong"), None, self.metric_box))
        gl.addWidget(self.group_plot, 1)
        gl.addWidget(label("Bars show the mean and standard error; each point is one video.", "caption"))

        body, lay = page(spacing=8)
        lay.addLayout(row(label("Results", "title"), None, self.clf_box,
                          button("Open in Excel", slot=lambda: self._open("results.xlsx")),
                          button("Open folder", slot=lambda: self._open(""))))
        lay.addWidget(self.headline)
        lay.addSpacing(20)
        lay.addWidget(label("Attack bouts", "strong"))
        lay.addWidget(self.ethogram, 3)
        lay.addSpacing(20)
        bottom = QHBoxLayout()
        bottom.setSpacing(40)
        tcol = QVBoxLayout()
        tcol.setSpacing(8)
        tcol.addWidget(label("Per video", "strong"))
        tcol.addWidget(self.table, 1)
        bottom.addLayout(tcol, 3)
        bottom.addWidget(self.group_box, 2)
        lay.addLayout(bottom, 4)
        QVBoxLayout(self).setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(body)

    def _open(self, rel: str) -> None:
        if self.ws:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.ws.results_dir / rel)))

    def set_workspace(self, ws: Workspace) -> None:
        self.ws = ws
        self.load()

    def has_results(self) -> bool:
        return len(self.summary) > 0

    def load(self) -> None:
        self.summary, self.bouts = pd.DataFrame(), pd.DataFrame()
        if self.ws is not None and (self.ws.results_dir / "summary.csv").exists():
            self.summary = pd.read_csv(self.ws.results_dir / "summary.csv", dtype={"group": str})
            self.bouts = pd.read_csv(self.ws.results_dir / "bouts.csv")
            if self.ws.manifest_path.exists():
                man = pd.read_csv(self.ws.manifest_path, dtype=str, keep_default_na=False)
                keep = set(man.video_name)
                self.summary = self.summary[self.summary.video.isin(keep)].copy()
                self.bouts = self.bouts[self.bouts.video.isin(keep)]
                groups = dict(zip(man.video_name, man["group"] if "group" in man.columns else [""] * len(man)))
                self.summary["group"] = self.summary.video.map(groups).fillna("")
        cur = self.clf_box.currentText()
        clfs = list(dict.fromkeys(self.summary.get("behavior", pd.Series(dtype=str))))
        self.clf_box.blockSignals(True)
        self.clf_box.clear()
        self.clf_box.addItems(clfs)
        if cur in clfs:
            self.clf_box.setCurrentText(cur)
        self.clf_box.blockSignals(False)
        self.clf_box.setVisible(len(clfs) > 1)
        self.redraw()

    def redraw(self) -> None:
        clf = self.clf_box.currentText()
        s = self.summary[self.summary.behavior == clf].copy() if len(self.summary) else pd.DataFrame()
        cols = ["video", "group", "latency_s", "episodes", "bouts", "total_s", "pct_time", "mean_bout_s", "longest_bout_s"]
        if len(s) and not s.group.fillna("").astype(bool).any():
            cols.remove("group")
        self.table_model.set(s[[c for c in cols if c in s.columns]] if len(s) else pd.DataFrame())
        self._headline(s, clf)
        self._ethogram(s, clf)
        self._group_plot(s)

    def _headline(self, s: pd.DataFrame, clf: str) -> None:
        if not len(s):
            self.headline.setText("No results yet. Run the analysis on the Videos page.")
            return
        hit = s[s.bouts > 0]
        name = clf.replace("_", " ").lower()
        parts = [f"{name.capitalize()} in {len(hit)} of {len(s)} videos."]
        if len(hit):
            parts.append(f"In those, median latency {fmt_s(hit.latency_s.median())}, {hit.episodes.median():g} episodes "
                         f"and {fmt_s(hit.total_s.median())} of {name}.")
        self.headline.setText(" ".join(parts))

    def _group_plot(self, s: pd.DataFrame) -> None:
        p = self.group_plot
        p.clear()
        groups = list(dict.fromkeys(s.group.fillna(""))) if len(s) and "group" in s.columns else []
        show = len(groups) > 1
        self.group_box.setVisible(show)
        if not show:
            return
        metric = self.metric_box.currentData()
        rng = np.random.default_rng(0)
        ticks = []
        for i, g in enumerate(groups):
            v = s.loc[s.group.fillna("") == g, metric].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            col = QColor(GROUP_COLOURS[i % len(GROUP_COLOURS)])
            ticks.append((i, f"{g or 'No group'} (n = {len(v)})"))
            if not len(v):
                continue
            mean = v.mean()
            fill = QColor(col)
            fill.setAlpha(50)
            p.addItem(pg.BarGraphItem(x=[i], height=[mean], width=0.5, brush=fill, pen=pg.mkPen(None)))
            if len(v) > 1:
                sem = v.std(ddof=1) / np.sqrt(len(v))
                p.addItem(pg.ErrorBarItem(x=np.array([i]), y=np.array([mean]), height=np.array([2 * sem]), beam=0.1,
                                          pen=pg.mkPen(C["text"], width=1)))
            p.addItem(pg.ScatterPlotItem(x=i + rng.uniform(-0.12, 0.12, len(v)), y=v, size=7, brush=col, pen=pg.mkPen(None)))
        p.getAxis("bottom").setTicks([ticks])
        p.setXRange(-0.6, len(groups) - 0.4, padding=0)
        p.getPlotItem().setLabel("left", METRICS[metric], color=C["muted"], **{"font-size": "12px"})

    def _ethogram(self, s: pd.DataFrame, clf: str) -> None:
        p = self.ethogram
        p.clear()
        self.etho_rows = []
        if not len(s):
            return
        has_groups = "group" in s.columns and s.group.fillna("").astype(bool).any()
        order = s.sort_values(["group", "video"] if has_groups else ["video"]).video.tolist()
        self.etho_rows = order
        rown = {v: i for i, v in enumerate(order)}
        sess = s.set_index("video").loc[order, "session_s"].to_numpy(dtype=float)
        p.addItem(pg.BarGraphItem(x0=np.zeros(len(order)), x1=sess, y=np.arange(len(order)), height=0.5,
                                  brush=QColor(C["track"]), pen=pg.mkPen(None)))
        b = self.bouts[(self.bouts.behavior == clf) & self.bouts.video.isin(rown)]
        if len(b):
            p.addItem(pg.BarGraphItem(x0=b.start_s.to_numpy(), x1=b.end_s.to_numpy(), y=b.video.map(rown).to_numpy(),
                                      height=0.5, brush=QColor(C["attack"]), pen=pg.mkPen(None)))
        p.getAxis("left").setTicks([[(i, v) for i, v in enumerate(order)]])
        p.getAxis("left").setPen(pg.mkPen(None))
        p.setYRange(-0.6, len(order) - 0.4, padding=0)
        p.setXRange(0, float(np.nanmax(sess)) if len(sess) else 300, padding=0.01)
        p.invertY(True)
        p.setMinimumHeight(min(70 + 28 * len(order), 400))

    def _ethogram_click(self, ev) -> None:
        if not self.etho_rows or not self.ethogram.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = self.ethogram.getPlotItem().vb.mapSceneToView(ev.scenePos())
        i = int(round(pt.y()))
        if 0 <= i < len(self.etho_rows):
            v = self.etho_rows[i]
            b = self.bouts[(self.bouts.video == v) & (self.bouts.behavior == self.clf_box.currentText())]
            hit = b[(b.start_s - 1 <= pt.x()) & (b.end_s + 1 >= pt.x())]
            self.review_requested.emit(v, float(hit.start_s.iloc[0]) if len(hit) else max(float(pt.x()), 0.0))

    def _table_review(self, idx) -> None:
        v = self.proxy.data(self.proxy.index(idx.row(), 0))
        b = self.bouts[(self.bouts.video == v) & (self.bouts.behavior == self.clf_box.currentText())]
        self.review_requested.emit(v, float(b.start_s.iloc[0]) if len(b) else 0.0)


# ---------------------------------------------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------------------------------------------
class ReviewPage(QWidget):
    def __init__(self):
        super().__init__()
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        from PySide6.QtMultimediaWidgets import QVideoWidget

        self.ws: Workspace | None = None
        self.trace = pd.DataFrame()
        self.bouts = pd.DataFrame()
        self.cur_bouts = pd.DataFrame()
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setMuted(True)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget()
        self.video.setStyleSheet("background: #000000;")
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self._position)
        self.player.playbackStateChanged.connect(self._state)

        self.video_box = QComboBox()
        self.video_box.setMinimumWidth(220)
        self.video_box.currentTextChanged.connect(self.open_video)
        self.clf_box = QComboBox()
        self.clf_box.currentTextChanged.connect(self._draw)
        self.play = button("Play", slot=lambda: self.player.pause() if self.player.isPlaying() else self.player.play())
        self.play.setMinimumWidth(72)
        prev_b = button("Previous bout", slot=lambda: self._jump(-1))
        next_b = button("Next bout", slot=lambda: self._jump(1))
        self.speed = QComboBox()
        for r in (0.25, 0.5, 1.0, 2.0):
            self.speed.addItem(f"{r:g}× speed", r)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.speed.currentData()))
        self.time = label("0:00.0", "muted")
        self.time.setMinimumWidth(56)

        self.plot = pg.PlotWidget()
        style_plot(self.plot)
        self.plot.setFixedHeight(110)
        self.plot.setYRange(0, 1.02, padding=0)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getAxis("left").setTicks([[(0, "0"), (0.5, "0.5"), (1, "1")]])
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(C["accent"], width=1.5))
        self.cursor.sigDragged.connect(lambda line_: self.player.setPosition(int(line_.value() * 1000)))
        self.plot.scene().sigMouseClicked.connect(self._plot_click)

        self.bout_list = QListWidget()
        self.bout_list.itemClicked.connect(lambda it: it.data(Qt.ItemDataRole.UserRole) is not None
                                           and self.seek(it.data(Qt.ItemDataRole.UserRole) - 1.0))
        self.bout_list.setFixedWidth(220)

        body, lay = page(spacing=12)
        lay.addLayout(row(label("Review", "title"), 16, self.video_box, self.clf_box, None))
        main = QHBoxLayout()
        main.setSpacing(32)
        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(self.video, 1)
        left.addLayout(row(self.play, prev_b, next_b, 12, self.time, None, self.speed))
        left.addWidget(self.plot)
        left.addWidget(label("Probability of attack from the model; shaded where it was scored as attack. Click the graph "
                             "to jump.", "caption"))
        main.addLayout(left, 1)
        side = QVBoxLayout()
        side.setSpacing(8)
        side.addWidget(label("Bouts", "strong"))
        side.addWidget(self.bout_list, 1)
        main.addLayout(side)
        lay.addLayout(main, 1)
        QVBoxLayout(self).setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(body)

    def set_workspace(self, ws: Workspace) -> None:
        self.ws = ws
        self.player.stop()
        self.load()

    def load(self) -> None:
        cur = self.video_box.currentText()
        self.video_box.blockSignals(True)
        self.video_box.clear()
        if self.ws is not None and (self.ws.results_dir / "bouts.csv").exists():
            self.bouts = pd.read_csv(self.ws.results_dir / "bouts.csv")
            names = sorted(p.stem for p in (self.ws.results_dir / "traces").glob("*.csv"))
            if self.ws.manifest_path.exists():
                keep = set(pd.read_csv(self.ws.manifest_path, dtype=str, keep_default_na=False)["video_name"])
                names = [n for n in names if n in keep]
            self.video_box.addItems(names)
        self.video_box.blockSignals(False)
        if cur and self.video_box.findText(cur) >= 0:
            self.video_box.setCurrentText(cur)
        elif self.video_box.count():
            self.open_video(self.video_box.currentText())

    def open_video(self, name: str) -> None:
        if not name or self.ws is None:
            return
        self.trace = pd.read_csv(self.ws.results_dir / "traces" / f"{name}.csv")
        clfs = [c[len("Probability_"):] for c in self.trace.columns if c.startswith("Probability_")]
        cur = self.clf_box.currentText()
        self.clf_box.blockSignals(True)
        self.clf_box.clear()
        self.clf_box.addItems(clfs)
        if cur in clfs:
            self.clf_box.setCurrentText(cur)
        self.clf_box.blockSignals(False)
        self.clf_box.setVisible(len(clfs) > 1)
        self.player.setSource(QUrl.fromLocalFile(str(self.ws.root / "data" / "processed_videos" / f"{name}.mp4")))
        self._draw()

    def _draw(self) -> None:
        p = self.plot
        p.clear()
        self.bout_list.clear()
        clf = self.clf_box.currentText()
        if not len(self.trace) or not clf:
            return
        t = self.trace.time_s.to_numpy()
        on = self.trace[clf].to_numpy().astype(float)
        p.addItem(pg.PlotDataItem(t, on * 1.02, pen=pg.mkPen(None), fillLevel=0, brush=QColor(209, 52, 56, 45)))
        p.addItem(pg.PlotDataItem(t, self.trace[f"Probability_{clf}"].to_numpy(), pen=pg.mkPen(C["text"], width=1)))
        p.addItem(self.cursor)
        p.setXRange(0, float(t[-1]) if len(t) else 1, padding=0)
        name = self.video_box.currentText()
        b = self.bouts[(self.bouts.video == name) & (self.bouts.behavior == clf)] if len(self.bouts) else pd.DataFrame()
        self.cur_bouts = b
        for _, r in b.iterrows():
            it = QListWidgetItem(f"{fmt_time(r.start_s)} to {fmt_time(r.end_s)}   ({r.duration_s:.1f} s)")
            it.setData(Qt.ItemDataRole.UserRole, float(r.start_s))
            self.bout_list.addItem(it)
        if not len(b):
            self.bout_list.addItem(QListWidgetItem("No bouts in this video"))

    def seek(self, t: float) -> None:
        self.player.setPosition(int(max(t, 0.0) * 1000))
        if not self.player.isPlaying():
            self.player.play()

    def show_at(self, name: str, t: float) -> None:
        if self.video_box.findText(name) < 0:
            self.load()
        if self.video_box.currentText() != name:
            self.video_box.setCurrentText(name)
        QTimer.singleShot(300, lambda: self.seek(t - 1.0))

    def _jump(self, step: int) -> None:
        if not len(self.cur_bouts):
            return
        now = self.player.position() / 1000.0 + 1.0
        starts = self.cur_bouts.start_s.to_numpy()
        nxt = starts[starts > now + 0.05] if step > 0 else starts[starts < now - 0.5]
        if len(nxt):
            self.seek((nxt[0] if step > 0 else nxt[-1]) - 1.0)

    def _position(self, ms: int) -> None:
        self.cursor.setValue(ms / 1000.0)
        self.time.setText(fmt_time(ms / 1000.0))

    def _state(self, _state) -> None:
        self.play.setText("Pause" if self.player.isPlaying() else "Play")

    def _plot_click(self, ev) -> None:
        if self.plot.sceneBoundingRect().contains(ev.scenePos()):
            self.player.setPosition(int(max(self.plot.getPlotItem().vb.mapSceneToView(ev.scenePos()).x(), 0) * 1000))


# ---------------------------------------------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, template: Path):
        super().__init__()
        self.setWindowTitle("Aggression Scorer")
        self.resize(1320, 880)
        self.settings = QSettings("aggression-classifier", "app")
        self.videos = VideosPage(self.settings, template)
        self.results = ResultsPage()
        self.review = ReviewPage()

        self.pages = QStackedWidget()
        self.nav = QButtonGroup(self)
        header = QHBoxLayout()
        header.setContentsMargins(MARGIN_X, 0, MARGIN_X, 0)
        header.setSpacing(0)
        header.addWidget(label("Aggression Scorer", "brand"))
        header.addSpacing(28)
        self.nav_buttons = []
        for i, (name, w) in enumerate((("Videos", self.videos), ("Results", self.results), ("Review", self.review))):
            self.pages.addWidget(w)
            b = button(name, "tab")
            b.setCheckable(True)
            self.nav.addButton(b, i)
            self.nav_buttons.append(b)
            header.addWidget(b)
        header.addStretch(1)
        self.nav.idClicked.connect(self.pages.setCurrentIndex)
        self.nav_buttons[0].setChecked(True)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(header)
        lay.addWidget(line())
        lay.addWidget(self.pages, 1)
        self.setCentralWidget(central)

        self.videos.workspace_changed.connect(self._workspace)
        self.videos.results_ready.connect(self._results)
        self.results.review_requested.connect(self._review)
        self._enable()
        last = self.settings.value("last_folder", "")
        if last and Path(last).exists() and (Path(last) / "aggression_analysis").exists():
            QTimer.singleShot(0, lambda: self.videos.open_folder(Path(last)))

    def _go(self, i: int) -> None:
        self.nav_buttons[i].setChecked(True)
        self.pages.setCurrentIndex(i)

    def _enable(self) -> None:
        has = self.results.has_results()
        for i in (1, 2):
            self.nav_buttons[i].setEnabled(has)
            self.nav_buttons[i].setToolTip("" if has else "Run the analysis first")

    def _workspace(self, ws: Workspace) -> None:
        self.results.set_workspace(ws)
        self.review.set_workspace(ws)
        self._enable()

    def _results(self) -> None:
        self.results.load()
        self.review.load()
        self._enable()
        self._go(1)

    def _review(self, name: str, t: float) -> None:
        self._go(2)
        self.review.show_at(name, t)

    def closeEvent(self, e):
        if self.videos.runner.running():
            if QMessageBox.question(self, "Quit", "The analysis is still running. Stop it and quit?") != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.videos.runner.stop()
        e.accept()


def make_app() -> QApplication:
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Aggression Scorer")
    app.setStyle("Fusion")
    style.pick_font()
    app.setFont(style.font(14))
    app.setStyleSheet(style.stylesheet())
    pg.setConfigOptions(antialias=True, background=C["bg"], foreground=C["muted"])
    return app


def main(config: str | None = None) -> int:
    app = make_app()
    win = MainWindow(Path(config).resolve() if config else TEMPLATE_CONFIG)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
