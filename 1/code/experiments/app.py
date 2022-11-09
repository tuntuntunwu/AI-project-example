import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QAction, QStackedWidget,\
    QGridLayout, QPushButton, QLabel, QTextBrowser, QMessageBox, QFileDialog, QDesktopWidget
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt

import function_one
import function_two


class DiyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        # 状态栏
        self.statusBar()

        # 功能Action
        single_rec_act = QAction(QIcon('./res/0.png'), '单字识别', self)
        single_rec_act.setShortcut('Ctrl+F')
        single_rec_act.setStatusTip('对单个文字进行识别')
        single_rec_act.triggered.connect(self.changeLayout)
        rubbing_ret_act = QAction(QIcon('./res/1.png'), '单字检索', self)
        rubbing_ret_act.setShortcut('Ctrl+S')
        rubbing_ret_act.setStatusTip('对单个文字图片进行检索')
        rubbing_ret_act.triggered.connect(self.changeLayout)

        # 菜单栏
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('功能')
        fileMenu.addAction(single_rec_act)
        fileMenu.addAction(rubbing_ret_act)

        # 工具栏
        toolbar = self.addToolBar('功能')
        toolbar.addAction(single_rec_act)
        toolbar.addAction(rubbing_ret_act)

        # 需要的变量
        self.path = ""
        self.color = QColor(255, 255, 255)
        # 初始化第一个功能布局：单字识别
        self.initLayout0()
        # 初始化第二个功能布局：拓片检索
        self.initLayout1()

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.layout0)
        self.stacked.addWidget(self.layout1)
        self.setCentralWidget(self.stacked)

        self.setGeometry(100, 100, 600, 450)
        self.setFixedSize(600, 450)
        self.setWindowTitle('金文')
        self.setWindowIcon(QIcon('./res/icon.jpg'))
        self.setStyleSheet("QWidget { font-family: Microsoft YaHei; }")
        self.center()
        self.show()

    def initLayout0(self):
        # 布局
        grid = QGridLayout()
        self.lf = QLabel("单字识别图片")
        self.lf.setAlignment(Qt.AlignCenter)
        self.lf.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        self.lf.setAcceptDrops(True)
        self.lf.dragEnterEvent = self.dragEnterEventChr
        self.lf.dropEvent = self.dropEventChr
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
        btn0.clicked.connect(self.selectChrImg)
        btn1.clicked.connect(self.predictChrImg)

    def selectChrImg(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片')
        if fname[0]:
            self.path = fname[0]
            self.statusBar().showMessage("图片路径: " + self.path)
            pixmap = QPixmap(self.path)
            scaled_pixmap = pixmap.scaled(int(self.lf.width()), int(
                self.lf.height()), aspectRatioMode=Qt.KeepAspectRatio)
            self.lf.setPixmap(scaled_pixmap)
            self.l0_c.setText("")
            self.l1_c.setText("")
            self.l2_c.setText("")
            self.l3_c.setText("")
            self.l4_c.setText("")

    # 鼠标拖入事件
    def dragEnterEventChr(self, evn):
        for url in evn.mimeData().urls():
            self.statusBar().showMessage("图片路径: " + str(url.toLocalFile()))
        # 执行鼠标放开事件
        evn.accept()  # 不执行鼠标放开事件：evn.ignore()
 
    # 鼠标放开事件
    def dropEventChr(self, evn):
        for url in evn.mimeData().urls():
            self.path = str(url.toLocalFile())
            pixmap = QPixmap(self.path)
            scaled_pixmap = pixmap.scaled(int(self.lf.width()), int(
                self.lf.height()), aspectRatioMode=Qt.KeepAspectRatio)
            self.lf.setPixmap(scaled_pixmap)
            self.l0_c.setText("")
            self.l1_c.setText("")
            self.l2_c.setText("")
            self.l3_c.setText("")
            self.l4_c.setText("")
    
    def predictChrImg(self):
        if self.path and os.path.exists(self.path):
            top5_classes_probs = function_one.characterRecognition(self.path,
                "./classes.json", "./jinwen_vgg11.pt")
            self.l0_c.setText(
                top5_classes_probs[0] + ': ' + str(top5_classes_probs[1]))
            self.l1_c.setText(
                top5_classes_probs[2] + ': ' + str(top5_classes_probs[3]))
            self.l2_c.setText(
                top5_classes_probs[4] + ': ' + str(top5_classes_probs[5]))
            self.l3_c.setText(
                top5_classes_probs[6] + ': ' + str(top5_classes_probs[7]))
            self.l4_c.setText(
                top5_classes_probs[8] + ': ' + str(top5_classes_probs[9]))
        else:
            QMessageBox.information(self, "Info.", "请先选择图片^_^")

    def initLayout1(self):
        # 布局
        grid = QGridLayout()
        self.ls = QLabel("单字检索图片")
        self.ls.setAlignment(Qt.AlignCenter)
        self.ls.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        self.ls.setAcceptDrops(True)
        self.ls.dragEnterEvent = self.dragEnterEventRub
        self.ls.dropEvent = self.dropEventRub
        btn0 = QPushButton("选择图片")
        btn0.setStyleSheet(
            "QPushButton { height: 50 }")
        btn1 = QPushButton("检索")
        btn1.setStyleSheet(
            "QPushButton { height: 50 }")
        self.l0_l = QLabel("Top-1")
        self.l0_l.setAlignment(Qt.AlignCenter)
        self.l0_r = QLabel()
        self.l0_r.setAlignment(Qt.AlignCenter)
        self.l0_r.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        self.l1_l = QLabel("Top-2")
        self.l1_l.setAlignment(Qt.AlignCenter)
        self.l1_r = QLabel()
        self.l1_r.setAlignment(Qt.AlignCenter)
        self.l1_r.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        self.l2_l = QLabel("Top-3")
        self.l2_l.setAlignment(Qt.AlignCenter)
        self.l2_r = QLabel()
        self.l2_r.setAlignment(Qt.AlignCenter)
        self.l2_r.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        self.l3_l = QLabel("Top-4")
        self.l3_l.setAlignment(Qt.AlignCenter)
        self.l3_r = QLabel()
        self.l3_r.setAlignment(Qt.AlignCenter)
        self.l3_r.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        self.l4_l = QLabel("Top-5")
        self.l4_l.setAlignment(Qt.AlignCenter)
        self.l4_r = QLabel()
        self.l4_r.setAlignment(Qt.AlignCenter)
        self.l4_r.setStyleSheet(
            "QLabel { background-color: %s }" % self.color.name())
        grid.addWidget(self.ls, 1, 1, 10, 1)
        grid.addWidget(self.l0_l, 1, 2)
        grid.addWidget(self.l0_r, 1, 3)
        grid.addWidget(self.l1_l, 2, 2)
        grid.addWidget(self.l1_r, 2, 3)
        grid.addWidget(self.l2_l, 3, 2)
        grid.addWidget(self.l2_r, 3, 3)
        grid.addWidget(self.l3_l, 4, 2)
        grid.addWidget(self.l3_r, 4, 3)
        grid.addWidget(self.l4_l, 5, 2)
        grid.addWidget(self.l4_r, 5, 3)
        grid.addWidget(btn0, 6, 3, 2, 1)
        grid.addWidget(btn1, 8, 3, 2, 1)
        grid.setColumnStretch(1, 10)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 5)
        grid.setColumnStretch(4, 4)
        self.layout1 = QWidget()
        self.layout1.setLayout(grid)
        # 事件
        btn0.clicked.connect(self.selectRubImg)
        btn1.clicked.connect(self.predictRubImg)
    
    def selectRubImg(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片')
        if fname[0]:
            self.path = fname[0]
            self.statusBar().showMessage("图片路径: " + self.path)
            pixmap = QPixmap(self.path)
            scaled_pixmap = pixmap.scaled(int(self.ls.width()), int(
                self.ls.height()), aspectRatioMode=Qt.KeepAspectRatio)
            self.ls.setPixmap(scaled_pixmap)
            self.l0_l.setText("Top-1")
            self.l0_r.setText("")
            self.l1_l.setText("Top-2")
            self.l1_r.setText("")
            self.l2_l.setText("Top-3")
            self.l2_r.setText("")
            self.l3_l.setText("Top-4")
            self.l3_r.setText("")
            self.l4_l.setText("Top-5")
            self.l4_r.setText("")
    
    def dragEnterEventRub(self, evn):
        for url in evn.mimeData().urls():
            self.statusBar().showMessage("图片路径: " + str(url.toLocalFile()))
        evn.accept()
 
    def dropEventRub(self, evn):
        for url in evn.mimeData().urls():
            self.path = str(url.toLocalFile())
            pixmap = QPixmap(self.path)
            scaled_pixmap = pixmap.scaled(int(self.ls.width()), int(
                self.ls.height()), aspectRatioMode=Qt.KeepAspectRatio)
            self.ls.setPixmap(scaled_pixmap)
            self.l0_l.setText("Top-1")
            self.l0_r.setText("")
            self.l1_l.setText("Top-2")
            self.l1_r.setText("")
            self.l2_l.setText("Top-3")
            self.l2_r.setText("")
            self.l3_l.setText("Top-4")
            self.l3_r.setText("")
            self.l4_l.setText("Top-5")
            self.l4_r.setText("")

    def predictRubImg(self):
        if self.path and os.path.exists(self.path):
            top5_classes_probs,others = function_two.query(self.path)
            print(top5_classes_probs)
            # top5_classes_probs = query(self.path)
            if len(top5_classes_probs) >=5:
                self.l0_r.setText(
                    top5_classes_probs[0][0])
                self.l1_r.setText(
                    top5_classes_probs[1][0])
                self.l2_r.setText(
                    top5_classes_probs[2][0])
                self.l3_r.setText(
                    top5_classes_probs[3][0])
                self.l4_r.setText(
                    top5_classes_probs[4][0])
            elif len(top5_classes_probs) is 4:
                self.l0_r.setText(
                    top5_classes_probs[0][0])
                self.l1_r.setText(
                    top5_classes_probs[1][0])
                self.l2_r.setText(
                    top5_classes_probs[2][0])
                self.l3_r.setText(
                    top5_classes_probs[3][0])
            elif len(top5_classes_probs) is 3:
                self.l0_r.setText(
                    top5_classes_probs[0][0])
                self.l1_r.setText(
                    top5_classes_probs[1][0])
                self.l2_r.setText(
                    top5_classes_probs[2][0])
            elif len(top5_classes_probs) is 2:
                self.l0_r.setText(
                    top5_classes_probs[0][0])
                self.l1_r.setText(
                    top5_classes_probs[1][0])
            elif len(top5_classes_probs) is 1:
                self.l0_r.setText(
                    top5_classes_probs[0][0])
            else:
                QMessageBox.information(self, "Info.", "数据库中没有该图片^_^")
        else:
            QMessageBox.information(self, "Info.", "请先选择图片^_^")

    def changeLayout(self):
        sender = self.sender()
        tmp_str = sender.text()
        self.statusBar().showMessage(tmp_str)
        if tmp_str == '单字识别':
            self.stacked.setCurrentIndex(0)
        if tmp_str == '单字检索':
            self.stacked.setCurrentIndex(1)

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
