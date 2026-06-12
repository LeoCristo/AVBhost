import sys
import os
import time
import threading
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
import numpy as np

from PyQt5.QtCore import pyqtSignal, QThread, Qt, QSize, QTimer, QVariantAnimation, QPropertyAnimation
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QBrush
from PyQt5.QtWidgets import QFrame
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QSplitter,
    QToolBar,
    QAction,
    QTextEdit,
    QSizePolicy,
    QDialog,
)


DEFAULT_MODEL_FILE = "best.pt"
DEFAULT_ICON_FILE = "app_icon.png"


def ensure_icon(path=DEFAULT_ICON_FILE):
    if os.path.exists(path):
        return
    # generate a simple PNG icon using OpenCV (circle + flame triangle)
    try:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # background
        img[:] = (30, 30, 30)
        # flame (triangle)
        pts = np.array([[128, 30], [60, 180], [196, 180]], np.int32)
        cv2.fillConvexPoly(img, pts, (0, 140, 255))
        pts2 = np.array([[128, 60], [88, 170], [168, 170]], np.int32)
        cv2.fillConvexPoly(img, pts2, (0, 60, 255))
        # circle border
        cv2.circle(img, (128, 128), 100, (200, 80, 0), 6)
        cv2.imwrite(path, img)
    except Exception:
        # silent fail; icon optional
        return


def ensure_action_icons(base_dir="."):
    # create simple distinct icons for actions if missing
    icons = {
        "icon_webcam.png": (50, (50, 200, 50)),
        "icon_raspberry.png": (51, (200, 100, 50)),
        "icon_open.png": (52, (50, 150, 200)),
        "icon_model.png": (53, (200, 50, 150)),
    }
    for name, (seed, color) in icons.items():
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            continue
        try:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:] = (40, 40, 40)
            # simple circle with color and a white small symbol
            cv2.circle(img, (32, 32), 24, color, -1)
            cv2.putText(img, chr((seed % 26) + 65), (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(path, img)
        except Exception:
            pass


def ensure_connection_icons(base_dir="."):
    # generate simple computer and raspberry icons if missing
    icons = {
        "icon_computer.png": (80, (60, 80, 100)),
        "icon_raspberry.png": (81, (160, 40, 80)),
    }
    for name, (seed, color) in icons.items():
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            continue
        try:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:] = (24, 26, 28)
            if name == "icon_computer.png":
                # draw monitor
                cv2.rectangle(img, (8, 12), (56, 40), (80, 90, 100), -1)
                cv2.rectangle(img, (14, 18), (50, 34), (12, 18, 24), -1)
                cv2.rectangle(img, (28, 40), (36, 46), (80, 90, 100), -1)
            else:
                # draw balloon/raspberry simple circle cluster
                cv2.circle(img, (32, 20), 12, (color[0], color[1], color[2]), -1)
                cv2.circle(img, (24, 28), 8, (color[0]-20, color[1]-10, color[2]-30), -1)
                cv2.circle(img, (40, 28), 8, (color[0]-10, color[1]-20, color[2]-10), -1)
                cv2.rectangle(img, (30, 34), (34, 48), (20, 18, 24), -1)
            cv2.imwrite(path, img)
        except Exception:
            pass


def find_camera_index(max_index: int = 4):
    for i in range(max_index):
        try:
            # prefer DirectShow on Windows to avoid other noisy backends
            try:
                if sys.platform.startswith("win"):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(i)
            except Exception:
                cap = cv2.VideoCapture(i)
            ok = False
            if cap.isOpened():
                # read a few frames to ensure a real, active camera
                frames = []
                for _ in range(3):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    frames.append(frame)
                    time.sleep(0.03)
                if len(frames) >= 2:
                    try:
                        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
                        stds = [float(g.std()) if g.size > 0 else 0.0 for g in grays]
                        # require reasonable noise/variation and at least two frames different
                        overall_std = sum(stds) / len(stds) if stds else 0.0
                        diffs = [float(np.mean(np.abs((grays[i].astype(float) - grays[i-1].astype(float))))) for i in range(1, len(grays))]
                        max_diff = max(diffs) if diffs else 0.0
                    except Exception:
                        overall_std = 0.0
                        max_diff = 0.0
                    ok = (overall_std >= 2.0) or (max_diff >= 1.0)
            try:
                cap.release()
            except Exception:
                pass
            if ok:
                return i
        except Exception:
            continue
    return None


def find_all_cameras(max_index: int = 6):
    found = []
    for i in range(max_index):
        try:
            try:
                if sys.platform.startswith("win"):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(i)
            except Exception:
                cap = cv2.VideoCapture(i)
            ok = False
            if cap.isOpened():
                # read a few frames and verify variation to avoid false positives
                frames = []
                for _ in range(3):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    frames.append(frame)
                    time.sleep(0.03)
                if len(frames) >= 2:
                    try:
                        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
                        stds = [float(g.std()) if g.size > 0 else 0.0 for g in grays]
                        overall_std = sum(stds) / len(stds) if stds else 0.0
                        diffs = [float(np.mean(np.abs((grays[i].astype(float) - grays[i-1].astype(float))))) for i in range(1, len(grays))]
                        max_diff = max(diffs) if diffs else 0.0
                    except Exception:
                        overall_std = 0.0
                        max_diff = 0.0
                    ok = (overall_std >= 2.0) or (max_diff >= 1.0)
            try:
                cap.release()
            except Exception:
                pass
            if ok:
                found.append(i)
        except Exception:
            pass
    return found


class VideoThread(QThread):
    frame_ready = pyqtSignal(object)
    started = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self._running = False

    def run(self):
        self._running = True
        # open capture using platform-appropriate backend when possible
        try:
            if isinstance(self.source, int) and sys.platform.startswith("win"):
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.source)
        except Exception:
            cap = cv2.VideoCapture(self.source)
        # verify capture opened
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            self.error.emit(f"Não foi possível abrir fonte: {self.source}")
            return
        else:
            try:
                self.started.emit()
            except Exception:
                pass
        while self._running:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_ready.emit(frame)
            time.sleep(0.02)
        cap.release()

    def stop(self):
        self._running = False
        self.wait()


class ModelLoader(QThread):
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            model = YOLO(self.path)
            self.loaded.emit(model)
        except Exception as e:
            self.error.emit(str(e))


class RaspberryDialog(QDialog):
    """Dialog to test and preview a Raspberry (RTSP/HTTP) stream before connecting."""
    def __init__(self, parent=None, initial_url=""):
        super().__init__(parent)
        self.setWindowTitle("Conectar Raspberry / Stream")
        self.resize(560, 360)
        self.parent = parent

        layout = QVBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("rtsp://... or http://... (jpg/png image endpoint)")
        if initial_url:
            self.url_input.setText(initial_url)

        btn_layout = QHBoxLayout()
        self.btn_test = QPushButton("Testar")
        self.btn_connect = QPushButton("Conectar")
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_connect.setEnabled(False)

        btn_layout.addWidget(self.btn_test)
        btn_layout.addWidget(self.btn_connect)
        btn_layout.addWidget(self.btn_cancel)

        self.status_label = QLabel("Status: aguardando teste")
        self.status_label.setStyleSheet("color:#d0d0d0;")

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(520, 260)
        self.preview_label.setStyleSheet("background:#0e0e0e; border:1px solid #333; border-radius:6px;")
        self.preview_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.url_input)
        layout.addLayout(btn_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.preview_label)
        self.setLayout(layout)

        self.btn_test.clicked.connect(self._on_test)
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_cancel.clicked.connect(self.reject)

    def _set_status(self, text, ok=False):
        self.status_label.setText(f"Status: {text}")
        if ok:
            self.status_label.setStyleSheet("color:#8fe27a;")
        else:
            self.status_label.setStyleSheet("color:#f08080;")

    def _on_test(self):
        url = self.url_input.text().strip()
        if not url:
            self._set_status("URL vazia")
            return
        self._set_status("Testando...")
        QApplication.processEvents()
        # try HTTP image first
        if url.lower().startswith("http") and any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
            try:
                import requests

                resp = requests.get(url, timeout=6)
                if resp.status_code == 200:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._show_frame(frame)
                        self._set_status("Imagem HTTP recebida", ok=True)
                        self.btn_connect.setEnabled(True)
                        return
                self._set_status(f"Falha HTTP: {resp.status_code}")
            except Exception as e:
                self._set_status(f"Erro HTTP: {e}")
            return

        # otherwise try video/rtsp
        try:
            # choose backend for platform if possible
            try:
                if sys.platform.startswith("win"):
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(url)
            except Exception:
                cap = cv2.VideoCapture(url)
            # attempt to read a couple frames
            frames = []
            for _ in range(3):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
                time.sleep(0.03)
            try:
                cap.release()
            except Exception:
                pass
            if frames:
                self._show_frame(frames[-1])
                self._set_status("Stream recebido", ok=True)
                self.btn_connect.setEnabled(True)
            else:
                self._set_status("Não foi possível ler frames do stream")
        except Exception as e:
            self._set_status(f"Erro: {e}")

    def _show_frame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(self.preview_label.size(), Qt.KeepAspectRatio)
            self.preview_label.setPixmap(pix)
        except Exception:
            pass

    def _on_connect(self):
        url = self.url_input.text().strip()
        if not url:
            return
        # persist and start
        try:
            import json
            with open(os.path.join(os.getcwd(), "cameras.json"), "w", encoding="utf-8") as f:
                json.dump({"preferred": url}, f)
        except Exception:
            pass
        # ask parent to start
        try:
            if self.parent:
                self.parent.start_video(url)
        except Exception:
            pass
        self.accept()


class ConnectionWidget(QFrame):
    """Draws a small diagram: computer <-- status --> raspberry (balloon) with animated status."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 56)
        self._state = 'disconnected'
        self._pulse = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(2 * 3.14159)
        self._anim.setDuration(900)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_anim_value)

        # line opacity animator for smooth transitions between states
        self._line_alpha = 120
        self._alpha_anim = QPropertyAnimation(self, b"line_alpha")
        self._alpha_anim.setDuration(400)

    def set_state(self, state: str):
        self._state = state
        if state == 'connecting':
            self._anim.start()
            # brighten the line while connecting
            self._alpha_anim.stop()
            self._alpha_anim.setStartValue(self._line_alpha)
            self._alpha_anim.setEndValue(200)
            self._alpha_anim.start()
        elif state == 'connected':
            self._anim.stop()
            # set solid green line
            self._alpha_anim.stop()
            self._alpha_anim.setStartValue(self._line_alpha)
            self._alpha_anim.setEndValue(255)
            self._alpha_anim.start()
        else:
            self._anim.stop()
            # dim the line when disconnected
            self._alpha_anim.stop()
            self._alpha_anim.setStartValue(self._line_alpha)
            self._alpha_anim.setEndValue(80)
            self._alpha_anim.start()
        self.update()

    def _on_tick(self):
        # legacy; kept for compatibility if needed
        self._pulse += 0.3
        if self._pulse > 3.14:
            self._pulse = 0.0
        self.update()

    def _on_anim_value(self, v):
        try:
            self._pulse = float(v)
        except Exception:
            self._pulse = 0.0
        self.update()

    def get_line_alpha(self):
        return self._line_alpha

    def set_line_alpha(self, v):
        self._line_alpha = int(v)
        self.update()

    line_alpha = property(get_line_alpha, set_line_alpha)

    def paintEvent(self, ev):
        w = self.width()
        h = self.height()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # background transparent
        painter.fillRect(0, 0, w, h, Qt.transparent)

        # left: computer icon (try image first)
        comp_w, comp_h = 48, 34
        comp_x, comp_y = 8, (h - comp_h) // 2
        try:
            comp_path = os.path.join(os.getcwd(), 'icon_computer.png')
            if os.path.exists(comp_path):
                pix = QPixmap(comp_path).scaled(comp_w, comp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(comp_x, comp_y, pix)
            else:
                painter.setBrush(QColor(50, 60, 70))
                painter.setPen(QColor(80, 90, 100))
                painter.drawRoundedRect(comp_x, comp_y, comp_w, comp_h, 4, 4)
                # screen
                painter.setBrush(QColor(20, 24, 28))
                painter.drawRect(comp_x + 6, comp_y + 6, comp_w - 12, comp_h - 12)
                # stand
                painter.setBrush(QColor(40, 48, 56))
                painter.drawRect(comp_x + (comp_w//2) - 6, comp_y + comp_h, 12, 6)
        except Exception:
            painter.setBrush(QColor(50, 60, 70))
            painter.setPen(QColor(80, 90, 100))
            painter.drawRoundedRect(comp_x, comp_y, comp_w, comp_h, 4, 4)

        # right: balloon raspberry camera (try image first)
        ball_w = 44
        ball_x = w - ball_w - 12
        ball_y = (h - 36) // 2
        try:
            rasp_path = os.path.join(os.getcwd(), 'icon_raspberry.png')
            if os.path.exists(rasp_path):
                pix = QPixmap(rasp_path).scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(ball_x, ball_y, pix)
            else:
                painter.setBrush(QColor(160, 40, 80))
                painter.setPen(QColor(130, 30, 60))
                painter.drawEllipse(ball_x, ball_y, 36, 36)
                # small camera lens
                painter.setBrush(QColor(20, 20, 24))
                painter.setPen(QColor(60, 60, 66))
                painter.drawEllipse(ball_x + 10, ball_y + 8, 12, 12)
                painter.setBrush(QColor(120, 120, 140))
                painter.drawEllipse(ball_x + 14, ball_y + 12, 4, 4)
                # tail
                painter.setPen(QColor(130, 30, 60))
                painter.drawLine(ball_x + 18, ball_y + 36, ball_x + 22, ball_y + 44)
        except Exception:
            painter.setBrush(QColor(160, 40, 80))
            painter.setPen(QColor(130, 30, 60))
            painter.drawEllipse(ball_x, ball_y, 36, 36)

        # center status symbol
        center_x = comp_x + comp_w + 24
        center_y = h // 2
        # draw a line between computer and raspberry
        # color depends on state: red (disconnected), yellow pulse (connecting), green (connected)
        if self._state == 'disconnected':
            line_color = QColor(200, 60, 60, self._line_alpha)
        elif self._state == 'connecting':
            glow = 120 + int((1 + np.sin(self._pulse)) * 60)
            line_color = QColor(250, 200, 30, min(255, glow))
        else:
            line_color = QColor(46, 125, 50, self._line_alpha)
        painter.setPen(line_color)
        painter.drawLine(comp_x + comp_w, center_y, ball_x, center_y)

        # status circle
        if self._state == 'disconnected':
            color = QColor(200, 60, 60)
            radius = 9
        elif self._state == 'connecting':
            # smooth pulsing for the center circle
            glow = 160 + int((1 + np.sin(self._pulse)) * 60)
            color = QColor(250, 200, 30, max(120, glow))
            radius = 10 + int((1 + np.sin(self._pulse)) * 3)
        else:
            # connected stable green with slight glow
            glow = 200
            color = QColor(46, 125, 50, glow)
            radius = 10

        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        # small status icon (like plug)
        painter.setPen(QColor(255, 255, 255, 220))
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(center_x - 6, center_y - 6, 12, 12, 30 * 16, 120 * 16)

        painter.end()

    def _set_status(self, text, ok=False):
        self.status_label.setText(f"Status: {text}")
        if ok:
            self.status_label.setStyleSheet("color:#8fe27a;")
        else:
            self.status_label.setStyleSheet("color:#f08080;")

    def _on_test(self):
        url = self.url_input.text().strip()
        if not url:
            self._set_status("URL vazia")
            return
        self._set_status("Testando...")
        QApplication.processEvents()
        # try HTTP image first
        if url.lower().startswith("http") and any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
            try:
                import requests

                resp = requests.get(url, timeout=6)
                if resp.status_code == 200:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._show_frame(frame)
                        self._set_status("Imagem HTTP recebida", ok=True)
                        self.btn_connect.setEnabled(True)
                        return
                self._set_status(f"Falha HTTP: {resp.status_code}")
            except Exception as e:
                self._set_status(f"Erro HTTP: {e}")
            return

        # otherwise try video/rtsp
        try:
            # choose backend for platform if possible
            try:
                if sys.platform.startswith("win"):
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(url)
            except Exception:
                cap = cv2.VideoCapture(url)
            # attempt to read a couple frames
            frames = []
            for _ in range(3):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
                time.sleep(0.03)
            try:
                cap.release()
            except Exception:
                pass
            if frames:
                self._show_frame(frames[-1])
                self._set_status("Stream recebido", ok=True)
                self.btn_connect.setEnabled(True)
            else:
                self._set_status("Não foi possível ler frames do stream")
        except Exception as e:
            self._set_status(f"Erro: {e}")

class MainWindow(QMainWindow):
    alert_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detecção de Queimadas - PyQt")
        # set window icon (generate if missing)
        try:
            ensure_icon()
            self.setWindowIcon(QIcon(DEFAULT_ICON_FILE))
            # ensure per-action icons exist
            ensure_action_icons()
            ensure_connection_icons()
        except Exception:
            pass
        self.model = None
        self.video_thread = None

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        # model name label in status bar
        self.model_label = QLabel("Modelo: (nenhum)")
        self.status.addPermanentWidget(self.model_label)

        # alert cooldown (seconds)
        self._last_alert = 0.0
        self._alert_cooldown = 5.0
        self.alert_signal.connect(self._show_alert)

        # HUD / stats
        self._last_frame_time = 0.0
        self._current_fps = 0.0
        self._last_frame_hash = None
        self._same_frame_count = 0
        self._same_frame_threshold = 12
        self._last_detection_key = None
        self._last_detection_time = 0.0

        self._ensure_model()

        self.image_label = QLabel("Carregue uma imagem ou inicie a câmera")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.btn_webcam = QPushButton("Webcam: selecionar/usar")
        self.btn_webcam.setToolTip("Clique para selecionar ou usar a webcam detectada")
        self.btn_raspberry = QPushButton("Stream / Raspberry (URL)")
        self.btn_raspberry.setToolTip("Abrir stream RTSP/HTTP ou endpoint de imagem")
        self.btn_open_file = QPushButton("Abrir imagem/vídeo")
        self.btn_open_file.setToolTip("Abrir arquivo de imagem ou vídeo para teste")
        self.btn_load_model = QPushButton("Carregar modelo (.pt)")
        self.btn_load_model.setToolTip("Carregar arquivo .pt customizado")
        self.btn_stop_cam = QPushButton("Parar fonte")
        self.btn_stop_cam.setToolTip("Parar captura/stream atual")
        self.btn_quit = QPushButton("Sair")
        self.btn_quit.setToolTip("Fechar o aplicativo")

        self.btn_webcam.clicked.connect(self.use_webcam)
        self.btn_raspberry.clicked.connect(self.use_raspberry)
        self.btn_open_file.clicked.connect(self.open_file)
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        self.btn_stop_cam.clicked.connect(self.stop_video)
        self.btn_quit.clicked.connect(self.close)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_webcam)
        controls.addWidget(self.btn_raspberry)
        controls.addWidget(self.btn_open_file)
        controls.addWidget(self.btn_load_model)
        controls.addWidget(self.btn_stop_cam)
        controls.addWidget(self.btn_quit)

        # logfile viewer
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumWidth(400)

        # alert area (replaces popup alerts) - shows last alert message
        self.alert_label = QLabel("Nenhum alerta")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setFixedHeight(60)
        self.alert_label.setWordWrap(True)
        self.alert_label.setStyleSheet("background-color: #2b2b2b; color: #dcdcdc; padding:10px; border-radius:8px; font-weight:600;")

        # main layout: left video + controls, right alerts+logs (splitter)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        # top bar: model info + camera indicator
        top_bar = QHBoxLayout()
        self.model_info = QLabel(self.model_label.text())
        self.model_info.setStyleSheet("font-weight:600; font-size:13px; color:#f0f0f0;")
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.cam_indicator = QLabel()
        self.cam_indicator.setStyleSheet("color:#fff; padding-left:8px;")
        self.cam_indicator.setContentsMargins(6, 0, 6, 0)
        # connection animation state
        self._cam_state = 'disconnected'  # 'disconnected' | 'connecting' | 'connected'
        self._cam_pulse = 0.0
        self._cam_anim_timer = QTimer(self)
        self._cam_anim_timer.setInterval(80)
        self._cam_anim_timer.timeout.connect(self._update_cam_anim)
        # initialize indicator
        self.set_cam_status('disconnected')
        top_bar.addWidget(self.model_info)
        # add visual connection widget
        try:
            self.conn_widget = ConnectionWidget()
            top_bar.addWidget(self.conn_widget)
        except Exception:
            top_bar.addWidget(spacer)
        top_bar.addWidget(spacer)
        top_bar.addWidget(self.cam_indicator)
        left_layout.addLayout(top_bar)
        # style the image area for a modern card look
        self.image_label.setStyleSheet("background:#111213; border-radius:12px; border:1px solid #333; padding:8px;")
        left_layout.addWidget(self.image_label)
        left_layout.addLayout(controls)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.alert_label)
        right_layout.addWidget(self.log_widget)
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # toolbar with small icons (reusing app_icon)
        toolbar = QToolBar("Main")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        # load per-action icons (fallback to default)
        def _icon(name):
            p = os.path.join(os.getcwd(), name)
            if os.path.exists(p):
                return QIcon(p)
            try:
                return QIcon(DEFAULT_ICON_FILE)
            except Exception:
                return QIcon()

        act_webcam = QAction(_icon("icon_webcam.png"), "Webcam", self)
        act_webcam.triggered.connect(self.use_webcam)
        self.act_webcam = act_webcam
        act_rpi = QAction(_icon("icon_raspberry.png"), "Raspberry", self)
        act_rpi.triggered.connect(self.use_raspberry)
        act_open = QAction(_icon("icon_open.png"), "Abrir", self)
        act_open.triggered.connect(self.open_file)
        act_model = QAction(_icon("icon_model.png"), "Carregar modelo", self)
        act_model.triggered.connect(self.load_model_dialog)

        toolbar.addAction(act_webcam)
        toolbar.addAction(act_rpi)
        toolbar.addAction(act_open)
        toolbar.addAction(act_model)
        # show text under icons for clarity
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        # detect cameras at startup and cache results to avoid rescanning on button click
        cams = find_all_cameras(max_index=6)
        self._detected_cams = cams
        if not cams:
            act_webcam.setEnabled(False)
            act_webcam.setToolTip("Nenhuma webcam detectada no sistema. Use 'Stream / Raspberry' para adicionar uma fonte.")
            # disable the quick webcam button as well
            try:
                self.btn_webcam.setEnabled(False)
            except Exception:
                pass
        else:
            cam_idx = cams[0]
            act_webcam.setToolTip(f"Webcam detectada: índice {cam_idx} — clique para selecionar")
            # store preferred camera index (default to first found)
            self._preferred_cam_index = cam_idx
        # load persisted cameras if any
        try:
            cfg = {}
            cfg_path = os.path.join(os.getcwd(), "cameras.json")
            if os.path.exists(cfg_path):
                import json
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            if cfg.get("preferred") is not None:
                self._preferred_cam_index = cfg.get("preferred")
                act_webcam.setEnabled(True)
                act_webcam.setToolTip(f"Usar webcam (índice {self._preferred_cam_index})")
        except Exception:
            pass

        self.log_file = open("events.log", "a", encoding="utf-8")
        # processing flag to avoid queuing frames
        self._processing = False

    def _ensure_model(self):
        # If default model file missing, try download from HF repo used in teste.py
        if not os.path.exists(DEFAULT_MODEL_FILE):
            self.status_message("Baixando modelo padrão (pode demorar)...")
            try:
                modelo = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
                os.replace(modelo, DEFAULT_MODEL_FILE)
            except Exception as e:
                self.status_message(f"Falha ao baixar modelo: {e}")
        try:
            self.status_message("Carregando modelo YOLO...")
            self.model = YOLO(DEFAULT_MODEL_FILE)
            self.status_message("Modelo carregado")
            try:
                self.model_label.setText(f"Modelo: {DEFAULT_MODEL_FILE}")
                if hasattr(self, 'model_info'):
                    self.model_info.setText(f"Modelo: {DEFAULT_MODEL_FILE}")
            except Exception:
                pass
        except Exception as e:
            self.status_message(f"Erro ao carregar modelo: {e}")

        # try load last used model path
        try:
            if os.path.exists("last_model.txt"):
                with open("last_model.txt", "r", encoding="utf-8") as f:
                    p = f.read().strip()
                if p and os.path.exists(p) and p != DEFAULT_MODEL_FILE:
                    try:
                        self.status_message("Carregando modelo personalizado...")
                        self.model = YOLO(p)
                        try:
                            self.model_label.setText(f"Modelo: {os.path.basename(p)}")
                            if hasattr(self, 'model_info'):
                                self.model_info.setText(f"Modelo: {os.path.basename(p)}")
                        except Exception:
                            pass
                        self.status_message("Modelo personalizado carregado")
                    except Exception:
                        pass
        except Exception:
            pass

    def status_message(self, msg: str):
        self.status.showMessage(msg, 5000)

    def open_image(self):
        # stop any running video source before showing a static image
        self.stop_video()
        path, _ = QFileDialog.getOpenFileName(self, "Abrir imagem", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        frame = cv2.imread(path)
        self.process_and_show(frame)

    def open_file(self):
        # stop any running video source before opening a file (image or video)
        self.stop_video()
        path, _ = QFileDialog.getOpenFileName(self, "Abrir arquivo", "", "Media (*.png *.jpg *.jpeg *.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            self.start_video(path)
        else:
            frame = cv2.imread(path)
            self.process_and_show(frame)

    def start_camera(self):
        self.start_video(0)

    def start_video(self, source=0):
        # stop existing
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        try:
            # If source is an integer (webcam index), check availability first
            try:
                src_int = int(source)
            except Exception:
                src_int = None
            if src_int is not None:
                try:
                    if sys.platform.startswith("win"):
                        cap_test = cv2.VideoCapture(src_int, cv2.CAP_DSHOW)
                    else:
                        cap_test = cv2.VideoCapture(src_int)
                except Exception:
                    cap_test = cv2.VideoCapture(src_int)
                ok = cap_test.isOpened()
                try:
                    cap_test.release()
                except Exception:
                    pass
                if not ok:
                    self.status_message("Nenhuma câmera encontrada no índice especificado.")
                    return

            self.video_thread = VideoThread(source=source)
            self.video_thread.frame_ready.connect(self.process_and_show)
            # on error, show message and mark disconnected
            self.video_thread.error.connect(lambda e: (self.status_message(e), self.set_cam_status('disconnected')))
            # when the thread reports started, set connected
            self.video_thread.started.connect(lambda: (self.status_message(f"Fonte iniciada: {source}"), self.set_cam_status('connected')))
            self.video_thread.start()
        except Exception as e:
            self.status_message(f"Falha ao iniciar fonte: {e}")

    def stop_camera(self):
        self.stop_video()

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            self.status_message("Fonte parada")
        # mark disconnected
        try:
            self.set_cam_status('disconnected')
        except Exception:
            pass

    def process_and_show(self, frame):
        if frame is None:
            return
        # drop frame if a previous frame is still being processed
        if getattr(self, "_processing", False):
            return
        self._processing = True
        orig = frame.copy()
        # detect identical/static frames (possible fake camera/no camera)
        try:
            # use a cheap hash: sum of a downsampled view
            small = orig[::16, ::16]
            frame_hash = int(small.sum())
            if self._last_frame_hash is not None and frame_hash == self._last_frame_hash:
                self._same_frame_count += 1
            else:
                self._same_frame_count = 0
            self._last_frame_hash = frame_hash
            if self._same_frame_count >= self._same_frame_threshold:
                # stop video source and warn user
                try:
                    self.stop_video()
                except Exception:
                    pass
                self.status_message("Fonte parada: frames estáticos (sem câmera?)")
                self._processing = False
                return
        except Exception:
            pass
        # FPS calculation (simple)
        try:
            now = time.time()
            if self._last_frame_time > 0:
                dt = now - self._last_frame_time
                if dt > 0:
                    self._current_fps = 0.9 * self._current_fps + 0.1 * (1.0 / dt) if self._current_fps > 0 else (1.0 / dt)
            self._last_frame_time = now
        except Exception:
            pass
        alert_triggered = False
        try:
            if self.model is not None:
                results = self.model(orig)
                if len(results) > 0:
                    r = results[0]
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        # Try to extract tensors/arrays safely
                        try:
                            xyxy = boxes.xyxy.cpu().numpy()
                        except Exception:
                            try:
                                xyxy = boxes.xyxy.numpy()
                            except Exception:
                                xyxy = []
                        try:
                            confs = boxes.conf.cpu().numpy()
                        except Exception:
                            try:
                                confs = boxes.conf.numpy()
                            except Exception:
                                confs = []
                        try:
                            cls_ids = boxes.cls.cpu().numpy()
                        except Exception:
                            try:
                                cls_ids = boxes.cls.numpy()
                            except Exception:
                                cls_ids = []

                        for i, coords in enumerate(xyxy if len(xyxy) else []):
                            x1, y1, x2, y2 = map(int, coords[:4])
                            conf = float(confs[i]) if i < len(confs) else 0.0
                            cls = int(cls_ids[i]) if i < len(cls_ids) else -1
                            label = self.model.names.get(cls, str(cls)) if hasattr(self.model, "names") else str(cls)
                            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(orig, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            if ("fire" in label.lower()) or ("smoke" in label.lower()):
                                alert_triggered = True
                                # dedupe similar logs within short time window
                                key = f"{label}:{int(conf*1000)}"
                                now = time.time()
                                if key != self._last_detection_key or (now - self._last_detection_time) > 3.0:
                                    self._log_event(label, conf)
                                    self._last_detection_key = key
                                    self._last_detection_time = now
        except Exception as e:
            # Non-fatal; show raw frame
            print("Detecção falhou:", e)

        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        # draw HUD: FPS and model name
        try:
            fps_text = f"FPS: {self._current_fps:.1f}"
            model_text = self.model_label.text() if hasattr(self, "model_label") else "Modelo: (nenhum)"
            cv2.putText(rgb, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(rgb, model_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        except Exception:
            pass
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio))

        if alert_triggered:
            self._alert_user()
        else:
            # no alerts: update status and reset alert area to neutral
            try:
                self.status_message("Nenhuma detecção")
            except Exception:
                pass
            try:
                if hasattr(self, "alert_label"):
                    self.alert_label.setText("Nenhum alerta")
                    self.alert_label.setStyleSheet("background-color: #3a3a3a; color: #eaeaea; padding:8px; border-radius:4px;")
            except Exception:
                pass
        # mark processing done
        try:
            self._processing = False
        except Exception:
            pass

    def load_model_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar modelo .pt", "", "PyTorch model (*.pt)")
        if not path:
            return
        # start loader thread
        self.status_message("Carregando modelo (isso pode demorar)...")
        self._pending_model_path = path
        self.model_loader = ModelLoader(path)
        self.model_loader.loaded.connect(self._on_model_loaded)
        self.model_loader.error.connect(self._on_model_error)
        self.model_loader.start()

    def _on_model_loaded(self, model_obj):
        # replace current model
        try:
            self.model = model_obj
            # persist path
            try:
                path = getattr(self, "_pending_model_path", None)
                if path:
                    with open("last_model.txt", "w", encoding="utf-8") as f:
                        f.write(path)
                    try:
                        self.model_label.setText(f"Modelo: {os.path.basename(path)}")
                        if hasattr(self, 'model_info'):
                            self.model_info.setText(f"Modelo: {os.path.basename(path)}")
                    except Exception:
                        pass
            except Exception:
                pass
            self.status_message("Modelo carregado com sucesso")
        except Exception as e:
            self.status_message(f"Erro ao aplicar modelo: {e}")

    def _on_model_error(self, err_msg):
        self.status_message(f"Falha ao carregar modelo: {err_msg}")

    def _alert_user(self):
        # Emit a signal to show the alert on the main (GUI) thread.
        now = time.time()
        if now - self._last_alert < self._alert_cooldown:
            return
        self._last_alert = now
        try:
            # send a clear descriptive message to alert area
            self.alert_signal.emit("Possível incêndio/fumaça detectado!")
        except Exception:
            pass

    def _show_alert(self, message: str):
        # update alert area instead of a modal popup
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            text = f"{ts} - {message}"
            # set prominent red background for a short time
            try:
                self.alert_label.setText(message)
                self.alert_label.setStyleSheet("background-color: #8b0000; color: #fff; padding:8px; border-radius:4px;")
            except Exception:
                pass
            # append to log as well
            try:
                if hasattr(self, "log_widget"):
                    self.log_widget.append(text)
            except Exception:
                pass
        except Exception:
            pass
    def set_cam_status(self, state: str):
        """Set camera connection state and update indicator.
        state: 'disconnected' | 'connecting' | 'connected'
        """
        try:
            self._cam_state = state
            if state == 'disconnected':
                self._cam_anim_timer.stop()
                pix = self._make_circle_pixmap(QColor(180, 60, 60), 14)
                self.cam_indicator.setPixmap(pix)
                self.cam_indicator.setToolTip('Desconectado')
                self.cam_indicator.setText('  Desconectado')
                try:
                    if hasattr(self, 'conn_widget'):
                        self.conn_widget.set_state('disconnected')
                except Exception:
                    pass
            elif state == 'connecting':
                # start pulsing animation
                self._cam_pulse = 0.0
                self._cam_anim_timer.start()
                self.cam_indicator.setToolTip('Conectando...')
                self.cam_indicator.setText('  Conectando')
                try:
                    if hasattr(self, 'conn_widget'):
                        self.conn_widget.set_state('connecting')
                except Exception:
                    pass
            elif state == 'connected':
                self._cam_anim_timer.stop()
                pix = self._make_circle_pixmap(QColor(46, 125, 50), 14)
                self.cam_indicator.setPixmap(pix)
                self.cam_indicator.setToolTip('Conectado')
                self.cam_indicator.setText('  Conectado')
                try:
                    if hasattr(self, 'conn_widget'):
                        self.conn_widget.set_state('connected')
                except Exception:
                    pass
            else:
                self._cam_anim_timer.stop()
        except Exception:
            pass

    def _update_cam_anim(self):
        # called by timer when in 'connecting' state; update pulse
        try:
            self._cam_pulse += 0.25
            if self._cam_pulse > 3.14:
                self._cam_pulse = 0.0
            # pulse between 8 and 16 px
            r = 10 + int((1 + np.sin(self._cam_pulse)) * 3)
            alpha = 180 + int((1 + np.sin(self._cam_pulse)) * 40)
            color = QColor(250, 200, 20, max(80, min(255, alpha)))
            pix = self._make_circle_pixmap(color, r)
            self.cam_indicator.setPixmap(pix)
        except Exception:
            pass

    def _make_circle_pixmap(self, qcolor: QColor, diameter: int = 12) -> QPixmap:
        size = max(12, diameter)
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        try:
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            brush = QBrush(qcolor)
            painter.setBrush(brush)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, size - 1, size - 1)
            painter.end()
        except Exception:
            pass
        return pix
    def use_webcam(self):
        # use cached detected cameras (do not rescan here) and include saved stream
        cams = getattr(self, "_detected_cams", [])
        choices = [f"Índice {c}" for c in cams]
        # include persisted URL if present
        try:
            cfg_path = os.path.join(os.getcwd(), "cameras.json")
            if os.path.exists(cfg_path):
                import json
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                pref = cfg.get("preferred")
                if pref and isinstance(pref, str) and not pref.isdigit():
                    choices.append(pref)
        except Exception:
            pass

        if not choices:
            self.status_message("Nenhuma câmera detectada no sistema.")
            return

        item, ok = QInputDialog.getItem(self, "Selecionar câmera", "Escolha uma fonte:", choices, 0, False)
        if not ok or not item:
            return

        # numeric selection
        if item.startswith("Índice"):
            try:
                idx = int(item.split()[1])
            except Exception:
                self.status_message("Índice de câmera inválido")
                return
            # final check before start
            cap = cv2.VideoCapture(idx)
            okp = cap.isOpened()
            try:
                cap.release()
            except Exception:
                pass
            if not okp:
                self.status_message(f"Índice {idx} não disponível")
                return
            self._preferred_cam_index = idx
            try:
                import json
                with open("cameras.json", "w", encoding="utf-8") as f:
                    json.dump({"preferred": idx}, f)
            except Exception:
                pass
            self.start_video(idx)
            return

        # otherwise URL
        url = item
        self._preferred_cam_index = url
        try:
            import json
            with open("cameras.json", "w", encoding="utf-8") as f:
                json.dump({"preferred": url}, f)
        except Exception:
            pass
        self.start_video(url)

    def add_camera_dialog(self):
        # Ask user for camera index or stream URL
        txt, ok = QInputDialog.getText(self, "Adicionar câmera", "Digite índice (0,1,...) ou URL (rtsp/http):")
        if not ok or not txt:
            return
        txt = txt.strip()
        # try integer first
        try:
            idx = int(txt)
            cap = cv2.VideoCapture(idx)
            okp = cap.isOpened()
            try:
                cap.release()
            except Exception:
                pass
            if not okp:
                self.status_message(f"Índice {idx} não retornou vídeo.")
                return
            # success: set preferred
            self._preferred_cam_index = idx
            try:
                import json
                with open("cameras.json", "w", encoding="utf-8") as f:
                    json.dump({"preferred": idx}, f)
            except Exception:
                pass
            self.act_webcam.setEnabled(True)
            self.act_webcam.setToolTip(f"Usar webcam (índice {idx})")
            self.status_message(f"Câmera adicionada: índice {idx}")
            return
        except Exception:
            pass
        # treat as URL
        url = txt
        try:
            cap = cv2.VideoCapture(url)
            okp = cap.isOpened()
            try:
                cap.release()
            except Exception:
                pass
            if not okp:
                self.status_message("Não foi possível abrir o stream/URL informado.")
                return
            # success
            self._preferred_cam_index = url
            try:
                import json
                with open("cameras.json", "w", encoding="utf-8") as f:
                    json.dump({"preferred": url}, f)
            except Exception:
                pass
            self.act_webcam.setEnabled(True)
            self.act_webcam.setToolTip("Usar webcam (stream configurado)")
            self.status_message("Câmera de stream adicionada")
        except Exception:
            self.status_message("Erro ao testar URL da câmera.")

    def select_camera_dialog(self):
        # Offer a list of detected camera indices for user to choose
        cams = find_all_cameras(max_index=6)
        choices = [str(c) for c in cams]
        # also include persisted preferred if it's a URL
        try:
            cfg_path = os.path.join(os.getcwd(), "cameras.json")
            if os.path.exists(cfg_path):
                import json
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                pref = cfg.get("preferred")
                if pref and isinstance(pref, str) and not pref.isdigit():
                    choices.append(pref)
        except Exception:
            pass
        if not choices:
            self.status_message("Nenhuma câmera detectada para selecionar.")
            return
        item, ok = QInputDialog.getItem(self, "Selecionar câmera", "Câmeras encontradas:", choices, 0, False)
        if not ok or not item:
            return
        # if selected is numeric index
        try:
            idx = int(item)
            self._preferred_cam_index = idx
            try:
                import json
                with open("cameras.json", "w", encoding="utf-8") as f:
                    json.dump({"preferred": idx}, f)
            except Exception:
                pass
            self.status_message(f"Câmera selecionada: índice {idx}")
            return
        except Exception:
            pass
        # otherwise assume URL
        url = item
        self._preferred_cam_index = url
        try:
            import json
            with open("cameras.json", "w", encoding="utf-8") as f:
                json.dump({"preferred": url}, f)
        except Exception:
            pass
        self.status_message("Câmera de stream selecionada")

    def use_raspberry(self):
        dlg = RaspberryDialog(parent=self, initial_url=getattr(self, '_preferred_cam_index', ''))
        res = dlg.exec_()
        # RaspberryDialog will call start_video on connect
        return

    def _log_event(self, label, conf):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} - {label} - {conf:.3f}\n"
        try:
            self.log_file.write(line)
            self.log_file.flush()
        except Exception:
            pass
        try:
            # append to on-screen log
            if hasattr(self, "log_widget"):
                self.log_widget.append(line.strip())
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            if self.video_thread:
                self.video_thread.stop()
        except Exception:
            pass
        try:
            self.log_file.close()
        except Exception:
            pass
        event.accept()

class EntryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bem-vindo - Detecção de Queimadas")
        self.setModal(True)
        self.resize(400, 260)
        layout = QVBoxLayout()
        # logo
        lbl_logo = QLabel()
        try:
            ensure_icon()
            pix = QPixmap(DEFAULT_ICON_FILE).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_logo.setPixmap(pix)
            lbl_logo.setAlignment(Qt.AlignCenter)
        except Exception:
            lbl_logo.setText("Detecção de Queimadas")
            lbl_logo.setAlignment(Qt.AlignCenter)

        lbl_title = QLabel("Sistema de Detecção de Queimadas")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px;")

        btn_start = QPushButton("Iniciar")
        btn_quit = QPushButton("Sair")
        btn_start.clicked.connect(self.accept)
        btn_quit.clicked.connect(self.reject)

        layout.addWidget(lbl_logo)
        layout.addWidget(lbl_title)
        layout.addStretch()
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_start)
        btn_layout.addWidget(btn_quit)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # dark theme stylesheet (simple)
    app.setStyleSheet("""
    QWidget { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1f2733, stop:1 #121417); color: #eaeaea; font-family: Segoe UI, Arial; }
    QMainWindow { background: transparent; }
    QStatusBar { background-color: rgba(20,20,20,0.6); color:#cfcfcf; }
    QPushButton { background-color: #2f3742; border: 1px solid #3d4450; padding: 8px 12px; border-radius:8px; color:#eaeaea; }
    QPushButton:hover { background-color: #3b4552; }
    QToolBar { background: transparent; border: none; spacing:6px; }
    QTextEdit { background-color: rgba(16,16,16,0.6); color: #e0e0e0; border-radius:8px; padding:8px; }
    QLabel { color: #eaeaea; }
    QSplitter::handle { background: rgba(255,255,255,0.02); }
    """)
    dlg = EntryDialog()
    res = dlg.exec_()
    if res == QDialog.Accepted:
        win = MainWindow()
        win.resize(900, 640)
        win.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)
