## from https://thesmithfam.org/blog/2010/03/10/fancy-qslider-stylesheet/
slider_style_sheet = '''
QSlider::groove:horizontal {
background-color: lightgray;
border: 1px solid #bbb;
height: 10px;
border-radius: 0px;
}

QSlider::sub-page:horizontal {
background-color: darkgray;
border: 1px solid #777;
height: 10px;
border-radius: 0px;
}

QSlider::add-page:horizontal {
border: 1px solid #777;
height: 10px;
border-radius: 0px;
}

QSlider::handle:horizontal {
border: 1px solid #777;
width: 13px;
margin-top: -1px;
margin-bottom: -1px;
border-radius: 0px;
}

QSlider::handle:horizontal:hover {
border: 1px solid #444;
border-radius: 0px;
}

QSlider::handle:horizontal:disabled {
border: 1px solid #aaa;
border-radius: 0px;
}
'''

ui_stylesheet = '''
QMainWindow {
font-size: 10px;
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
