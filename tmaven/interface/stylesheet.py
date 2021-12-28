## from https://thesmithfam.org/blog/2010/03/10/fancy-qslider-stylesheet/
slider_style_sheet = '''
QSlider::groove:horizontal {
border: 1px solid #bbb;
background: white;
height: 10px;
border-radius: 0px;
}

QSlider::sub-page:horizontal {
background: #ededff;
border: 1px solid #777;
height: 10px;
border-radius: 0px;
}

QSlider::add-page:horizontal {
background: #fff;
border: 1px solid #777;
height: 10px;
border-radius: 0px;
}

QSlider::handle:horizontal {
background: #ededff;
border: 1px solid #777;
width: 13px;
margin-top: -1px;
margin-bottom: -1px;
border-radius: 0px;
}

QSlider::handle:horizontal:hover {
background: #ddd;
border: 1px solid #444;
border-radius: 0px;
}

QSlider::sub-page:horizontal:disabled {
background: #bbb;
border-color: #999;
}

QSlider::add-page:horizontal:disabled {
background: #eee;
border-color: #999;
}

QSlider::handle:horizontal:disabled {
background: #eee;
border: 1px solid #aaa;
border-radius: 0px;
}
'''

ui_stylesheet = '''
QMainWindow {
color:black;
background-color:white;
font-size: 12px;
}
'''

ss_qmenubar = '''
QMenuBar {
	background-color: white;
}


QMenuBar::item:selected {
	background: #7da4ed;
}

QMenuBar::item:pressed {
	background: #5b667a;
}
'''
ss_qmenu = '''
QMenu {
	background-color: white;
}

QMenu::item:selected {
	background: #7da4ed;
}
QMenu::item:pressed {
	background: #5b667a;
}

'''


pad = 8
ss_plaintext = '''
QPlainTextEdit {
margin-left:%d;
margin-right:%d;
margin-top:%d;
margin-bottom:%d;
}
'''%(pad,pad,pad,pad)


ss_qdockwidget = '''
QDockWidget {border: 0px solid white}
QDockWidget::title {
	text-align: left;
	background: #ededff;
	padding-left: 35px;
}
'''

ss_qtableview = '''


QTableView {
	selection-color: black;
	selection-background-color: #ddd;
}
QHeaderView::section {
	border: 1px solid #777;
	border-radius: 0px;
}
QHeaderView::section:horizontal {
	background: #ededff;
	height: 15px;
}
QHeaderView::section:vertical {
	background: #ededff;
	height: 8px;
}
'''
