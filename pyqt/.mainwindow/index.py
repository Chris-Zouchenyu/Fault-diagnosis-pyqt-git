import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
# 导入login.py、main.py里面全部内容
from window1 import *
from window2 import *
from PyQt5.QtCore import Qt
# class Win_Login:
#     def __init__(self):
#         self.ui = QUiLoader().load(login.ui)
# class main(main.Ui_MainWindow, QMainWindow):
#     def __init__(self):
#         super(main, self).__init__()
#         self.setupUi(self)  # 初始化
#         self.pushButton_1.clicked.connect(self.display)
#     def display(self):
#         self.label_3.resize(400, 300)  # 重设Label大小
#         self.label_3.setScaledContents(True)  # 设置图片自适应窗口大小
#         self.label_3.setPixmap(QtGui.QPixmap("CO2.png"))
#   class login(login.Ui_Form, QMainWindow):
#     def __init__(self):
#         super(login, self).__init__()
#         self.setupUi(self)
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # 支持高分屏自动缩放
    app = QApplication(sys.argv)
    # 为main_window类和login_window类创建对象
    main_window = MainWindow()
    login_window = Form()
    # 显示登陆窗口
    login_window.show()
    # 将显示main_window与单击登录页面按钮绑定
    if login_window.on_pushButton_clicked:
        username = login_window.on_lineEdit_textEdited()
        passward = login_window.on_lineEdit_2_textEdited()
        if username == 'Chris' and passward == '123':
            login_window.close()
            main_window.show()
        else:
            print('no')
    # main_window.pushButton_2.clicked.connect(main_window.close)
    # 关闭程序，释放资源
    sys.exit(app.exec_())