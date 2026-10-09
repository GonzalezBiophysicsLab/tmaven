"""Apply desktop palettes to toolbars, including cached Matplotlib icons."""
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QWidget


def apply_toolbar_theme(toolbar, palette):
    toolbar.setBackgroundRole(QPalette.Window)
    toolbar.setForegroundRole(QPalette.WindowText)
    toolbar.setPalette(palette)
    for widget in toolbar.findChildren(QWidget):
        widget.setPalette(palette)
    toolbar.setStyleSheet('''
        QToolBar { border: none; background: palette(window);
                   color: palette(window-text); spacing: 0px; }
        QToolBar::separator { background: palette(mid); }
        QToolButton { color: palette(button-text); }
        QToolButton:hover, QToolButton:checked {
            background: palette(highlight); color: palette(highlighted-text);
        }
        QToolTip { background: palette(tool-tip-base); color: palette(tool-tip-text); }
    ''')
    # Matplotlib chooses icon colors from the palette when creating a QIcon.
    # A palette change doesn't recolor the QIcons already held by its actions.
    icon_factory = getattr(toolbar, '_icon', None)
    actions = getattr(toolbar, '_actions', {})
    if callable(icon_factory):
        for text, tooltip, image_file, callback in toolbar.toolitems:
            if image_file and callback in actions:
                actions[callback].setIcon(icon_factory(image_file + '.png'))
    toolbar.update()
