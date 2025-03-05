import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from MLP import *

# 设置全局字体
plt.rcParams['font.family'] = 'SimHei'  # 设置字体为黑体
plt.rcParams['font.size'] = 12  # 设置字体大小

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

data = pd.read_excel('D:\python\Deep learning\Fault diagnosis\pyqt\data.xlsx',header=None)
data = np.array(data)
dataset = []
for i in range(0,len(data),1024):
    dataset.append(data[i:i+1024])
dataset = torch.tensor(np.array(dataset),dtype=torch.float32)
dataset = dataset.reshape([-1,1024])
# print(dataset.shape)
# print(dataset)
for i in range(dataset.shape[0]):
            plt.figure(figsize=(12,8),dpi=200)
            plt.title('预测数据' + str(i+1))
            plt.plot([x for x in range(1024)],dataset[i])
            plt.show()
my_model = torch.load(r'D:\python\Deep learning\Fault diagnosis\n50_lr0.001_MLPNet',weights_only=False)
with torch.no_grad():
    y = my_model(dataset)
    # print(y)
    for data in y:
        label = torch.argmax(data)
        label = int(label)
        print(FAULT_DICT_NEW[label])



