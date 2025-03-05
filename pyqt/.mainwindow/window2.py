# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
"""
Module implementing MainWindow.
"""
# pyqt
from PyQt5.QtCore import pyqtSlot,QThread
from PyQt5.QtWidgets import QMainWindow,QWidget,QFileDialog,QApplication
from PyQt5.QtGui import QPixmap
from Ui_window2 import Ui_MainWindow

#sklearn数据处理
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#pytorch
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
import torchkeras

# 函数
from CNN import CNNnet
from MLP import MLPnet
from LSTMnet import LSTM_model
from ResNet import ResNet18

# 设置全局字体
plt.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
plt.rcParams['font.size'] = 12  # 设置字体大小

#故障数据标签
FAULT_LABEL_DICT = {'97': 0,
                    '105': 1,
                    '118': 2,
                    '130': 3,
                    '169': 4,
                    '185': 5,
                    '197': 6,
                    '209': 7,
                    '222': 8,
                    '234': 9}

#选取驱动端的数据进行建模
AXIS = '_DE_time'
#随机数种子
seed = 102
np.random.seed(seed)

class Thread_dataload(QThread):
    '''
    线程1
    '''
    def __init__(self):
        super().__init__()

    def run(self):
        self.dataload()

    def dataload(self):
        data_dir = r'D:\python\Deep learning\Fault diagnosis\data'
        time_steps = 1024
        window = 128
        noise = True
        snr = -10
        feature, label = [], []
        for fault_type in FAULT_LABEL_DICT:
            lab = FAULT_LABEL_DICT[fault_type]
            totalaxis = 'X' + fault_type + AXIS
            if fault_type == '97':
                totalaxis = 'X0' + fault_type + AXIS
            #加载并解析mat文件
            mat_data = loadmat(data_dir + '\\' + fault_type + '.mat')[totalaxis]
            #start, end = 0, self.time_steps
            #每隔self.time_steps窗口构建一个样本，指定样本之间重叠的数目
            for i in range(0, len(mat_data) - time_steps, window):# 这里time_steps: 1024  window: 128
                sub_mat_data = mat_data[i: (i+time_steps)].reshape(-1,)
                #是否往数据中添加噪声
                if noise:
                    sub_mat_data = self.awgn(sub_mat_data, snr)
                feature.append(sub_mat_data)
                label.append(lab)
        return np.array(feature, dtype='float32'), np.array(label, dtype="int64")

    def awgn(self, data, snr, seed=102):
        """
        添加高斯白噪声
        """
        np.random.seed(seed)
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(data ** 2) / len(data)
        npower = xpower / snr
        noise = np.random.randn(len(data)) * np.sqrt(npower)
        return np.array(data + noise)

class Thread_split(QThread):
    '''
    线程2
    '''
    def __init__(self, data_dir, time_steps=1024, window=128, mode='train', val_rate=0.3, test_rate=0.5, \
                 noise=False, snr=None, network='MLP'):
        super().__init__()
        self.time_steps = time_steps
        self.mode = mode
        self.noise = noise
        self.snr = snr
        self.network = network
        self.feature_all, self.label_all = self.transform(data_dir)
        self.window = window
        self.val_rate = val_rate
        self.test_rate = test_rate
    
    def run(self):
        self.split()
    
    def split(self):
        #训练集和验证集的划分
        train_feature, val_feature, train_label, val_label = \
        train_test_split(self.feature_all, self.label_all, test_size=self.val_rate, random_state=seed)
        #标准化
        train_feature, val_feature = self.standardization(train_feature, val_feature)
        #验证集和测试集的划分
        val_feature, test_feature, val_label, test_label = \
        train_test_split(val_feature, val_label, test_size=self.test_rate, random_state=seed)
        if self.mode == 'train':
            self.feature = train_feature
            self.label = train_label
        elif self.mode == 'val':
            self.feature = val_feature
            self.label = val_label
        elif self.mode == 'test':
            self.feature = test_feature
            self.label = test_label
        else:
            raise Exception("mode can only be one of ['train', 'val', 'test']")

    def transform(self, data_dir) :
        """
        转换函数,获取数据
        """
        feature, label = [], []
        for fault_type in FAULT_LABEL_DICT:
            lab = FAULT_LABEL_DICT[fault_type]
            totalaxis = 'X' + fault_type + AXIS
            if fault_type == '97':
                totalaxis = 'X0' + fault_type + AXIS
            #加载并解析mat文件
            mat_data = loadmat(data_dir + '\\' + fault_type + '.mat')[totalaxis]
            #start, end = 0, self.time_steps
            #每隔self.time_steps窗口构建一个样本，指定样本之间重叠的数目
            for i in range(0, len(mat_data) - self.time_steps, 128):# 这里time_steps: 1024  window: 128
                sub_mat_data = mat_data[i: (i+self.time_steps)].reshape(-1,)
                #是否往数据中添加噪声
                if self.noise:
                    sub_mat_data = self.awgn(sub_mat_data, self.snr)
                feature.append(sub_mat_data)
                label.append(lab)
        return np.array(feature, dtype='float32'), np.array(label, dtype="int64")
    
    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.feature)

    def awgn(self, data, snr, seed=seed):
        """
        添加高斯白噪声
        """
        np.random.seed(seed)
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(data ** 2) / len(data)
        npower = xpower / snr
        noise = np.random.randn(len(data)) * np.sqrt(npower)
        return np.array(data + noise)
    
    def standardization(self, train_data, val_data):
        """
        标准化
        """
        scalar = preprocessing.StandardScaler().fit(train_data)
        train_data = scalar.transform(train_data)
        val_data = scalar.transform(val_data)
        return train_data, val_data

class Thread_train(QThread):
    '''
    线程3
    '''
    def __init__(self,lr,batch_size,epoch,num_classes,network,x,y,path):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.num_classes = num_classes
        self.network = network
        self.x = x
        self.y = y
        self.path = path
    
    def run(self):
        self.train()
    
    def train(self):
        '''
        用于代码训练
        '''
        lr = self.lr
        batch_size = self.batch_size
        epoch = self.epoch
        num_classes = self.num_classes
        network = self.network
        x = self.x
        y = self.y
        path = self.path
        #torchkeras训练方式
        x = torch.tensor(x)
        y = torch.tensor(y)
        if network == 'MLPNet':
            mymodel = torchkeras.Model(MLPnet())
        elif network == 'CNNNet':
            mymodel = torchkeras.Model(CNNnet(num_classes))
        elif network == 'LSTMNet':
            mymodel = torchkeras.Model(LSTM_model(1,num_classes))
            x = torch.reshape(x,(x.shape[0],x.shape[1],1))
            y = torch.reshape(y,(y.shape[0],1))
        elif network == 'ResNet':
            mymodel = torchkeras.Model(ResNet18(num_classes))
        ds_train = TensorDataset(x,y)
        dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0)
        #优化器
        optimizer = torch.optim.SGD(mymodel.parameters(),lr=lr)
        #损失函数
        loss_fn = torch.nn.CrossEntropyLoss()
        mymodel.compile(loss_func=loss_fn, optimizer=optimizer)
        history = mymodel.fit(epochs=epoch,dl_train=dl_train)
        torch.save(mymodel,path)
        return history


class MainWindow(QMainWindow, Ui_MainWindow):
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
        self.data_all = False
        self.network = 'MLP'
    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Slot documentation goes here.
        """
        self.textBrowser.append('请选择数据')
        # 打开文件对话框，选择图片文件
        file_path = []
        folder_path = r'D:\python\Deep learning\Fault diagnosis\pyqt\picture'
        for filename in os.listdir(folder_path):
            # 拼接文件的完整路径
            file_path.append(os.path.join(folder_path, filename))
        self.load_image(file_path[0])  # 加载并显示图片
        self.i = 0
        self.textBrowser.append('数据选择成功')

    def load_image(self, image_path):
        """
        加载图片并显示在QLabel中

        @param image_path 图片文件路径
        @type str
        """
        pixmap = QPixmap(image_path)  # 创建QPixmap对象
        if not pixmap.isNull():  # 检查图片是否加载成功
            self.label_5.setPixmap(pixmap)  # 在label_5中显示图片
            self.label_5.setScaledContents(True)  # 让图片自适应QLabel的大小
        else:
            print("图片加载失败，请检查文件路径或格式。")    

    @pyqtSlot()
    def on_all_data_clicked(self):
        """
        Slot documentation goes here.
        """
        self.data_all = True

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.textBrowser.append('数据导入中')
        if self.data_all == True:
            self.thread1 = Thread_dataload()
            self.thread1.start()
            # self.xx,self.yy = self.thread1.dataload()
            self.textBrowser.append('数据导入成功')
        else:
            self.textBrowser.append('未勾选数据')

    @pyqtSlot()
    def on_radioButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.network = 'MLPNet'
    @pyqtSlot()
    def on_radioButton_2_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.network = 'CNNNet'
    @pyqtSlot()
    def on_radioButton_3_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.network = 'ResNet'
    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        self.textBrowser.clear()
        self.textBrowser.append('你好！欢迎使用本软件！')

    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        """
        Slot documentation goes here.
        """
        self.i += 1
        if self.i >= 10:
            self.i = 0
        file_path = []
        folder_path = r'D:\python\Deep learning\Fault diagnosis\pyqt\picture'
        for filename in os.listdir(folder_path):
            # 拼接文件的完整路径
            file_path.append(os.path.join(folder_path, filename))
        self.load_image(file_path[self.i])  # 加载并显示图片
        self.textBrowser.append('数据选择成功')

    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        try:
            val_rate = float(self.lineEdit.text())
            batch_size = int(self.lineEdit_2.text())
            epoch = int(self.lineEdit_3.text())
            lr = float(self.lineEdit_4.text())
        except ValueError:
            self.textBrowser.append("输入的值不是有效的数字，请重新输入！")
        network = self.network
        path = r'D:\python\Deep learning\Fault diagnosis\n50_lr0.001_pyqt_' + network
        self.thread2 = Thread_split('D:\python\Deep learning\Fault diagnosis\data',val_rate=val_rate)
        x,y = self.thread2.transform('D:\python\Deep learning\Fault diagnosis\data')
        self.textBrowser.append('训练开始')          
        self.thread3 = Thread_train(lr=lr,batch_size=batch_size,num_classes=10,network=network,x = x,y = y, path = path,epoch=epoch)
        self.thread3.run()
        self.textBrowser.append('训练结束')

    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Xlsx Files (*.xlsx)")
        data = pd.read_excel(file_path,header=None)
        data = np.array(data)
        dataset = []
        for i in range(0,len(data),1024):
            dataset.append(data[i:i+1024])
        dataset = torch.tensor(np.array(dataset),dtype=torch.float32)
        dataset = dataset.reshape([-1,1024])
        self.dataset_pred = dataset
    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        dataset = self.dataset_pred
        image_path = {}
        for i in range(dataset.shape[0]):
            plt.figure(figsize=(12,6),dpi=200)
            plt.title('预测数据' + str(i+1))
            plt.plot([x for x in range(1024)],dataset[i])
            image_path[i] = r'D:\python\Deep learning\Fault diagnosis\pyqt' + r'\fig_' + str(i+1) + r'.jpg'
            plt.savefig(image_path[i])
        self.image_path = image_path
        pixmap = QPixmap(image_path[0])  # 创建QPixmap对象
        self.ii = 1
        if not pixmap.isNull():  # 检查图片是否加载成功
            self.label_6.setPixmap(pixmap)  # 在label_5中显示图片
            self.label_6.setScaledContents(True)
        else:
            self.textBrowser_3.append('图片显示失败，请重新选择图片')

    @pyqtSlot(str)
    def on_comboBox_activated(self, p0):
        """
        Slot documentation goes here.

        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yet
        self.model_select = p0

    @pyqtSlot()
    def on_pushButton_8_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        try:
            FAULT_DICT_NEW = {
                        0: '正常数据',
                        1: '7密耳 内圈故障',
                        2: '7密耳 滚动体故障',
                        3: '7密耳 外圈故障',
                        4: '14密耳 内圈故障',
                        5: '14密耳 滚动体故障',
                        6: '14密耳 外圈故障',
                        7: '21密耳 内圈故障',
                        8: '21密耳 滚动体故障',
                        9: '21密耳 外圈故障'
            }
            # 模型选择部分
            dataset = self.dataset_pred
            if self.model_select == 'MLP':
                my_model = torch.load(r'D:\python\Deep learning\Fault diagnosis\n50_lr0.001_MLPNet',weights_only=False)
            elif self.model_select == 'CNN':
                my_model = torch.load(r'D:\python\Deep learning\Fault diagnosis\n50_lr0.001_CNNNet',weights_only=False)
            elif self.model_select == 'ResNet':   
                my_model = torch.load(r'D:\python\Deep learning\Fault diagnosis\n50_lr0.001_ResNet',weights_only=False)
            with torch.no_grad():
                y = my_model(dataset)
                # print(y)
                for data in y:
                    label = torch.argmax(data)
                    label = int(label)
                    print(FAULT_DICT_NEW[label])
                    self.textBrowser_3.append('\n--------------------------------------\n故障类型：\n' + FAULT_DICT_NEW[label])
        except:
            self.textBrowser_3.append('请选择数据，选择模型，谢谢\n--------------------------------------\n')

    @pyqtSlot()
    def on_pushButton_9_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        image_path = self.image_path
        ii = self.ii
        if ii >= len(image_path):
            pixmap = QPixmap(image_path[0])  # 创建QPixmap对象
            if not pixmap.isNull():  # 检查图片是否加载成功
                self.label_6.setPixmap(pixmap)  # 在label_5中显示图片
                self.label_6.setScaledContents(True)
            else:
                self.textBrowser_3.append('图片显示失败，请重新选择图片')
            self.ii = 0
        else:
            pixmap = QPixmap(image_path[ii])  # 创建QPixmap对象
            if not pixmap.isNull():  # 检查图片是否加载成功
                self.label_6.setPixmap(pixmap)  # 在label_5中显示图片
                self.label_6.setScaledContents(True)
            else:
                self.textBrowser_3.append('图片显示失败，请重新选择图片')
        self.ii += 1
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())

