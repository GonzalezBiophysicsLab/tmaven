"""A small startup splash with the application logo and loading label."""
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import QSplashScreen

from .resources import load_pixmap


def loading_splash(app):
    # Paint at the screen's pixel ratio to keep the logo sharp on Retina displays.
    screen = app.primaryScreen()
    ratio = screen.devicePixelRatio() if screen else 1
    image = QPixmap(int(320 * ratio), int(240 * ratio))
    image.setDevicePixelRatio(ratio)
    image.fill(app.palette().color(QPalette.Window))
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        logo = load_pixmap('logo.png')
        if not logo.isNull():
            logo = logo.scaled(int(144 * ratio), int(144 * ratio),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setDevicePixelRatio(ratio)
            width, height = logo.width() / ratio, logo.height() / ratio
            painter.drawPixmap(QRect(int((320-width)/2), int(24+(144-height)/2),
                                     int(width), int(height)), logo)
        painter.setPen(app.palette().color(QPalette.WindowText))
        painter.setFont(app.font())
        painter.drawText(QRect(16, 180, 288, 40), Qt.AlignCenter, 'Loading tMAVEN')
    finally:
        painter.end()
    return QSplashScreen(image, Qt.WindowStaysOnTopHint)
