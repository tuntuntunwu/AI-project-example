import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QAction, QStackedWidget,\
    QGridLayout, QPushButton, QLabel, QTextBrowser, QMessageBox, QFileDialog, QDesktopWidget
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt

import jinwen_predict
import get_info


class DiyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        # 状态栏
        self.statusBar()

        # 功能Action
        single_rec_act = QAction(QIcon('./res/0.png'), '文字识别', self)
        single_rec_act.setShortcut('Ctrl+F')
        single_rec_act.setStatusTip('对单个文字进行识别')
        single_rec_act.triggered.connect(self.changeLayout)
        single_info_act = QAction(QIcon('./res/1_1.png'), '单字信息', self)
        single_info_act.setShortcut('Ctrl+S')
        single_info_act.setStatusTip('单个文字的信息')
        single_info_act.triggered.connect(self.changeLayout)
        rubbing_info_act = QAction(QIcon('./res/2_1.png'), '拓片信息', self)
        rubbing_info_act.setShortcut('Ctrl+R')
        rubbing_info_act.setStatusTip('整个拓片的信息')
        rubbing_info_act.triggered.connect(self.changeLayout)

        # 菜单栏
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('功能')
        fileMenu.addAction(single_rec_act)
        fileMenu.addAction(single_info_act)
        fileMenu.addAction(rubbing_info_act)

        # 工具栏
        toolbar = self.addToolBar('功能')
        toolbar.addAction(single_rec_act)
        toolbar.addAction(single_info_act)
        toolbar.addAction(rubbing_info_act)

        # 需要的变量
        self.path = ""
        self.color = QColor(255, 255, 255)
        # 初始化第一个功能布局：文字识别
        self.initLayout0()
        # 初始化第二个功能布局：单字信息
        self.initLayout1()
        # 初始化第三个功能布局：拓片信息
        self.initLayout2()

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.layout0)
        self.stacked.addWidget(self.layout1)
        self.stacked.addWidget(self.layout2)
        self.setCentralWidget(self.stacked)

        self.setGeometry(100, 100, 600, 450)
        self.setFixedSize(600, 450)
        self.setWindowTitle('金文识别')
        self.setWindowIcon(QIcon('./res/4.png'))
        self.setStyleSheet("QWidget { font-family: Microsoft YaHei; }")
        self.center()
        self.show()

    def initLayout0(self):
        # 布局
        grid = QGridLayout()
        self.lf = QLabel("金文图片")
        self.lf.setAlignment(Qt.AlignCenter)
        self.lf.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        btn0 = QPushButton("选择图片")
        btn0.setStyleSheet(
            "QPushButton { height: 50 }")
        btn1 = QPushButton("识别")
        btn1.setStyleSheet(
            "QPushButton { height: 50 }")
        l0_l = QLabel("Top-1")
        l0_l.setAlignment(Qt.AlignCenter)
        self.l0_c = QLabel()
        self.l0_c.setAlignment(Qt.AlignCenter)
        self.l0_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        l1_l = QLabel("Top-2")
        l1_l.setAlignment(Qt.AlignCenter)
        self.l1_c = QLabel()
        self.l1_c.setAlignment(Qt.AlignCenter)
        self.l1_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        l2_l = QLabel("Top-3")
        l2_l.setAlignment(Qt.AlignCenter)
        self.l2_c = QLabel()
        self.l2_c.setAlignment(Qt.AlignCenter)
        self.l2_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        l3_l = QLabel("Top-4")
        l3_l.setAlignment(Qt.AlignCenter)
        self.l3_c = QLabel()
        self.l3_c.setAlignment(Qt.AlignCenter)
        self.l3_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        l4_l = QLabel("Top-5")
        l4_l.setAlignment(Qt.AlignCenter)
        self.l4_c = QLabel()
        self.l4_c.setAlignment(Qt.AlignCenter)
        self.l4_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        grid.addWidget(self.lf, 1, 1, 10, 1)
        grid.addWidget(l0_l, 1, 2)
        grid.addWidget(self.l0_c, 1, 3)
        grid.addWidget(l1_l, 2, 2)
        grid.addWidget(self.l1_c, 2, 3)
        grid.addWidget(l2_l, 3, 2)
        grid.addWidget(self.l2_c, 3, 3)
        grid.addWidget(l3_l, 4, 2)
        grid.addWidget(self.l3_c, 4, 3)
        grid.addWidget(l4_l, 5, 2)
        grid.addWidget(self.l4_c, 5, 3)
        grid.addWidget(btn0, 6, 3, 2, 1)
        grid.addWidget(btn1, 8, 3, 2, 1)
        grid.setColumnStretch(1, 10)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 5)
        grid.setColumnStretch(4, 4)
        self.layout0 = QWidget()
        self.layout0.setLayout(grid)
        # 事件
        btn0.clicked.connect(self.selectImg)
        btn1.clicked.connect(self.predictImg)

    def selectImg(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片')
        if fname[0]:
            self.path = fname[0]
            self.statusBar().showMessage("图片路径: " + self.path)
            pixmap = QPixmap(self.path)
            scaled_pixmap = pixmap.scaled(int(self.lf.width()), int(
                self.lf.height()), aspectRatioMode=Qt.KeepAspectRatio)
            self.lf.setPixmap(scaled_pixmap)

    def predictImg(self):
        if self.path and os.path.exists(self.path):
            top5_classes_probs = jinwen_predict.jinwenPredict(self.path)
            self.l0_c.setText(
                top5_classes_probs[0] + ': ' + top5_classes_probs[5] + '%')
            self.l1_c.setText(
                top5_classes_probs[1] + ': ' + top5_classes_probs[6] + '%')
            self.l2_c.setText(
                top5_classes_probs[2] + ': ' + top5_classes_probs[7] + '%')
            self.l3_c.setText(
                top5_classes_probs[3] + ': ' + top5_classes_probs[8] + '%')
            self.l4_c.setText(
                top5_classes_probs[4] + ': ' + top5_classes_probs[9] + '%')
        else:
            QMessageBox.information(self, "Info.", "请先选择图片^_^")

    def initLayout1(self):
        # 布局
        grid = QGridLayout()
        self.ls = QLabel("单字图片")
        self.ls.setAlignment(Qt.AlignCenter)
        self.ls.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        btn = QPushButton("选择单字图片显示信息")
        btn.setStyleSheet(
            "QPushButton { height: 50 }")
        l5_l = QLabel("新统计字用头")
        l5_l.setAlignment(Qt.AlignCenter)
        self.l5_c = QLabel()
        self.l5_c.setAlignment(Qt.AlignCenter)
        self.l5_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        l6_l = QLabel("字形出处")
        l6_l.setAlignment(Qt.AlignCenter)
        self.l6_c = QLabel()
        self.l6_c.setAlignment(Qt.AlignCenter)
        self.l6_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        l7_l = QLabel("器名")
        l7_l.setAlignment(Qt.AlignCenter)
        self.l7_c = QLabel()
        self.l7_c.setAlignment(Qt.AlignCenter)
        self.l7_c.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        l8_l = QLabel("时代")
        l8_l.setAlignment(Qt.AlignCenter)
        self.l8_c = QLabel()
        self.l8_c.setAlignment(Qt.AlignCenter)
        self.l8_c.setStyleSheet(
            "QLabel { background-color: %s; }" % self.color.name())
        l9_l = QLabel("释文")
        l9_l.setAlignment(Qt.AlignCenter)
        self.l9_c = QTextBrowser()
        self.l9_c.setStyleSheet(
            "QTextBrowser { background-color: %s; border: none }" % self.color.name())
        grid.addWidget(self.ls, 1, 1, 10, 1)
        grid.addWidget(l5_l, 1, 2)
        grid.addWidget(self.l5_c, 1, 3)
        grid.addWidget(l6_l, 2, 2)
        grid.addWidget(self.l6_c, 2, 3)
        grid.addWidget(l7_l, 3, 2)
        grid.addWidget(self.l7_c, 3, 3)
        grid.addWidget(l8_l, 4, 2)
        grid.addWidget(self.l8_c, 4, 3)
        grid.addWidget(l9_l, 5, 2)
        grid.addWidget(self.l9_c, 5, 3, 4, 1)
        grid.addWidget(btn, 9, 3, 2, 1)
        grid.setColumnStretch(1, 10)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 6)
        grid.setColumnStretch(4, 2)
        for i in range(1, 11):
            grid.setRowStretch(i, 1)
        self.layout1 = QWidget()
        self.layout1.setLayout(grid)
        # 事件
        btn.clicked.connect(self.showSingleInfo)

    def showSingleInfo(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片')
        if fname[0]:
            self.path = fname[0]
            self.statusBar().showMessage("图片路径: " + self.path)
            pixmap = QPixmap(self.path)
            scaled_pixmap = pixmap.scaled(int(self.ls.width()), int(
                self.ls.height()), aspectRatioMode=Qt.KeepAspectRatio)
            self.ls.setPixmap(scaled_pixmap)

            try:
                rows = get_info.singleInfo(self.path)
                if rows[0][0] == 's' or rows[0][0] == 'S':
                    self.l5_c.setText(rows[0][1])
                    self.l5_c.setStyleSheet(
                        "QLabel { font-family: SimSun; }")
                else:
                    self.l5_c.setText(rows[0])
                    self.l5_c.setStyleSheet(
                        "QLabel { font-family: jinwen_B; }")
                self.l6_c.setText(rows[1])
                self.l7_c.setText(rows[2])
                self.l8_c.setText(rows[3])
                self.l9_c.setText(rows[4])
            except TypeError:
                self.l5_c.setText("")
                self.l6_c.setText("")
                self.l7_c.setText("")
                self.l8_c.setText("")
                self.l9_c.setText("")
                QMessageBox.information(self, "Info.", "数据库中未找到该图片信息^_^")

    def initLayout2(self):
        # 布局
        grid = QGridLayout()
        self.lr = QLabel("拓片图片")
        self.lr.setAlignment(Qt.AlignCenter)
        self.lr.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        btn = QPushButton("选择拓片显示释文")
        btn.setStyleSheet(
            "QPushButton { height: 50 }")
        l10_l = QLabel("释文")
        l10_l.setAlignment(Qt.AlignCenter)
        self.l10_c = QTextBrowser()
        self.l10_c.setStyleSheet(
            "QTextBrowser { background-color: %s; border: none }" % self.color.name())
        grid.addWidget(self.lr, 1, 1, 10, 1)
        grid.addWidget(l10_l, 1, 2)
        grid.addWidget(self.l10_c, 2, 2, 7, 1)
        grid.addWidget(btn, 9, 2, 2, 1)
        grid.setColumnStretch(1, 10)
        grid.setColumnStretch(2, 8)
        grid.setColumnStretch(3, 2)
        self.layout2 = QWidget()
        self.layout2.setLayout(grid)
        # 事件
        btn.clicked.connect(self.showRubbingInfo)

    def showRubbingInfo(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片')
        if fname[0]:
            self.path = fname[0]
            self.statusBar().showMessage("图片路径: " + self.path)
            pixmap = QPixmap(self.path)
            scaled_pixmap = pixmap.scaled(int(self.lr.width()), int(
                self.lr.height()), aspectRatioMode=Qt.KeepAspectRatio)
            self.lr.setPixmap(scaled_pixmap)

            rows = get_info.rubbingInfo(self.path)
            if rows:
                self.l10_c.setText(rows[0])
            else:
                self.l10_c.setText("")
                QMessageBox.information(self, "Info.", "数据库中未找到该图片信息^_^")

    def changeLayout(self):
        sender = self.sender()
        tmp_str = sender.text()
        self.statusBar().showMessage(tmp_str)
        if tmp_str == '文字识别':
            self.stacked.setCurrentIndex(0)
        if tmp_str == '单字信息':
            self.stacked.setCurrentIndex(1)
        if tmp_str == '拓片信息':
            self.stacked.setCurrentIndex(2)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = DiyWindow()
    sys.exit(app.exec_())
