# -*- coding: utf-8 -*-

"""
Module implementing Form.
"""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget,QApplication

from Ui_window1 import Ui_Form
from window2 import *

class Form(QWidget, Ui_Form):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget (defaults to None)
        @type QWidget (optional)
        """
        super().__init__(parent)
        self.setupUi(self)

    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()
        # 判断用户名和密码
        if username == "Chris" and password == "123456":  # 密码是 123456
            self.ui2 = MainWindow()  # 创建 MainWindow 对象
            self.ui1 = Form()
            self.ui2.show()  # 显示 MainWindow
            # self.ui1.close()
        else:
            print("用户名或密码错误")
    @pyqtSlot(str)
    def on_lineEdit_textEdited(self, p0):
        """
        Slot documentation goes here.

        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yet
        username = p0
        return username
    @pyqtSlot(str)
    def on_lineEdit_2_textEdited(self, p0):
        """
        Slot documentation goes here.

        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yeta
        password = p0
        return password
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)# 命令行
    ui = Form()
    ui.show()
    sys.exit(app.exec()) 