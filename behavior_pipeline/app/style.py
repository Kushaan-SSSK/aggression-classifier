# Kushaan Sharma
"""Look of the desktop app, following Microsoft's Windows 11 / Fluent 2 guidance:
- one font: Segoe UI Variable, the Windows system font (Segoe UI on Windows 10)
- the Windows type ramp: 12 / 14 / 20 / 28 px; regular for text, semibold only for titles
- neutral greys with one accent colour, 4 px corners on controls
- sections separated by space, not boxes or lines
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy

HERE = Path(__file__).resolve().parent

# Fluent 2 neutral and brand tokens
C = {"bg": "#FFFFFF", "text": "#242424", "muted": "#616161", "faint": "#8A8A8A", "stroke": "#D1D1D1", "line": "#E0E0E0",
     "hover": "#F5F5F5", "pressed": "#E0E0E0", "select": "#EBF3FC", "accent": "#0F6CBD", "accent_hover": "#115EA3",
     "attack": "#D13438", "warn": "#BC4B09", "track": "#F0F0F0"}
GROUP_COLOURS = ["#0F6CBD", "#D13438", "#107C10", "#CA5010", "#8764B8", "#038387", "#616161"]
FAMILY = "Segoe UI"
MONO = '"Cascadia Mono", Menlo, "DejaVu Sans Mono"'


def pick_font() -> str:
    """The system's own interface font: Segoe UI Variable on Windows 11, Segoe UI on Windows 10, and whatever the
    desktop uses on macOS (San Francisco) and Linux."""
    global FAMILY
    fams = QFontDatabase.families()
    if "Segoe UI Variable" in fams:
        FAMILY = "Segoe UI Variable"
    elif "Segoe UI" in fams:
        FAMILY = "Segoe UI"
    else:
        FAMILY = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    return FAMILY


def font(px: int = 14, semibold: bool = False) -> QFont:
    f = QFont(FAMILY)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight.DemiBold if semibold else QFont.Weight.Normal)
    return f


def _asset(name: str, svg: str) -> str:
    p = HERE / "assets" / name
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        p.write_text(svg, encoding="utf-8")
    return p.as_posix()


def stylesheet() -> str:
    tick = _asset("check.svg", '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14"><path d="M3 7.2l2.6 2.6L11 4.4" '
                               'fill="none" stroke="#fff" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    chevron = _asset("chevron.svg", '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12"><path d="M3 4.5l3 3 3-3" '
                                    'fill="none" stroke="#616161" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return f"""
* {{ font-family: "{FAMILY}"; }}
QWidget {{ background: {C['bg']}; color: {C['text']}; font-size: 14px; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 28px; font-weight: 600; }}
QLabel#subtitle {{ font-size: 20px; font-weight: 600; }}
QLabel#strong {{ font-weight: 600; }}
QLabel#muted {{ color: {C['muted']}; }}
QLabel#caption {{ color: {C['muted']}; font-size: 12px; }}
QLabel#brand {{ font-weight: 600; }}
QFrame#line {{ background: {C['line']}; border: none; min-height: 1px; max-height: 1px; }}

QPushButton {{ background: {C['bg']}; color: {C['text']}; border: 1px solid {C['stroke']}; border-radius: 4px;
               padding: 0 12px; min-height: 30px; }}
QPushButton:hover {{ background: {C['hover']}; }}
QPushButton:pressed {{ background: {C['pressed']}; }}
QPushButton:disabled {{ color: #BDBDBD; border-color: {C['line']}; }}
QPushButton#primary {{ background: {C['accent']}; color: #FFFFFF; border: 1px solid {C['accent']}; font-weight: 600; }}
QPushButton#primary:hover {{ background: {C['accent_hover']}; border-color: {C['accent_hover']}; }}
QPushButton#primary:disabled {{ background: #E0E0E0; border-color: #E0E0E0; color: #BDBDBD; }}
QPushButton#tab {{ border: none; border-bottom: 2px solid transparent; border-radius: 0; background: transparent;
                   color: {C['muted']}; padding: 0 2px; min-height: 46px; margin: 0 12px; }}
QPushButton#tab:hover {{ color: {C['text']}; }}
QPushButton#tab:checked {{ color: {C['text']}; font-weight: 600; border-bottom: 2px solid {C['accent']}; }}
QPushButton#tab:disabled {{ color: #C8C8C8; }}

QTableView {{ border: none; gridline-color: transparent; selection-background-color: {C['select']};
              selection-color: {C['text']}; outline: 0; alternate-background-color: #FAFAFA; }}
QTableView::item {{ padding: 0 8px; }}
QHeaderView {{ border: none; background: {C['bg']}; }}
QHeaderView::section {{ background: {C['bg']}; color: {C['muted']}; border: none; border-bottom: 1px solid {C['line']};
                        padding: 6px 8px; font-size: 12px; }}
QTableCornerButton::section {{ background: {C['bg']}; border: none; }}

QListWidget {{ border: none; outline: 0; }}
QListWidget::item {{ padding: 6px 4px; }}
QListWidget::item:hover {{ background: {C['hover']}; }}
QListWidget::item:selected {{ background: {C['select']}; color: {C['text']}; }}

QLineEdit, QComboBox {{ border: 1px solid {C['stroke']}; border-bottom-color: #A8A8A8; border-radius: 4px; padding: 0 8px;
                        min-height: 30px; background: {C['bg']}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({chevron}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ border: 1px solid {C['stroke']}; selection-background-color: {C['select']};
                               selection-color: {C['text']}; outline: 0; }}
QPlainTextEdit {{ border: 1px solid {C['line']}; border-radius: 4px; font-family: {MONO}, Consolas, monospace; font-size: 12px;
                  color: {C['muted']}; background: #FAFAFA; }}

QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QTableView::indicator {{ width: 16px; height: 16px; border: 1px solid #8A8A8A; border-radius: 3px;
                                               background: #FFFFFF; }}
QCheckBox::indicator:checked, QTableView::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']};
                                                               image: url({tick}); }}

QSlider {{ background: transparent; min-height: 20px; }}
QSlider::groove:horizontal {{ height: 4px; background: #D1D1D1; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: {C['accent']};
                              border: 2px solid #FFFFFF; }}

QProgressBar {{ border: none; background: {C['line']}; border-radius: 1px; min-height: 3px; max-height: 3px; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 1px; }}

QScrollBar:vertical {{ width: 10px; background: transparent; margin: 0; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; margin: 0; }}
QScrollBar::handle:vertical {{ background: #C8C8C8; border-radius: 3px; min-height: 30px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #C8C8C8; border-radius: 3px; min-width: 30px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QToolTip {{ background: #FFFFFF; color: {C['text']}; border: 1px solid {C['stroke']}; padding: 4px 8px; }}
"""


# ---- small widgets -----------------------------------------------------------------------------------------------
def label(text: str = "", kind: str = "", wrap: bool = False) -> QLabel:
    w = QLabel(text)
    if kind:
        w.setObjectName(kind)
    w.setWordWrap(wrap)
    return w


def button(text: str, kind: str = "", slot=None, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if slot is not None:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


def line() -> QFrame:
    f = QFrame()
    f.setObjectName("line")
    return f


class ElidedLabel(QLabel):
    """A one-line label that shortens long text with … in the middle instead of widening the window."""

    def __init__(self, text: str = "", kind: str = ""):
        super().__init__()
        if kind:
            self.setObjectName(kind)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 (Qt name)
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide()

    def _elide(self) -> None:
        QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, max(self.width(), 50)))


def style_plot(pw, left: str = "", bottom: str = "") -> None:
    """White, frameless pyqtgraph plot with light axes and grey 12 px tick labels."""
    import pyqtgraph as pg

    pw.setBackground(C["bg"])
    pi = pw.getPlotItem()
    pi.hideButtons()
    pi.setMenuEnabled(False)
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen("#D1D1D1"))
        ax.setTextPen(pg.mkPen(C["muted"]))
        ax.setStyle(tickFont=font(12), tickLength=-4, tickTextOffset=6)
    label_style = {"color": C["muted"], "font-size": "12px", "font-family": FAMILY}
    if left:
        pi.setLabel("left", left, **label_style)
    if bottom:
        pi.setLabel("bottom", bottom, **label_style)
