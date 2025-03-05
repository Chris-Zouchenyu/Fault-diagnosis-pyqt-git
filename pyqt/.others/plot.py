from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

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
FAULT_DICT = {
                    '97': '正常数据',
                    '105': '7密耳 内圈故障',
                    '118': '7密耳 滚动体故障',
                    '130': '7密耳 外圈故障',
                    '169': '14密耳 内圈故障',
                    '185': '14密耳 滚动体故障',
                    '197': '14密耳 外圈故障',
                    '209': '21密耳 内圈故障',
                    '222': '21密耳 滚动体故障',
                    '234': '21密耳 外圈故障'
}
#选取驱动端的数据进行建模
AXIS = '_DE_time'

def awgn(data, snr, seed=102):
        """
        添加高斯白噪声
        """
        np.random.seed(seed)
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(data ** 2) / len(data)
        npower = xpower / snr
        noise = np.random.randn(len(data)) * np.sqrt(npower)
        return np.array(data + noise)

data_dir = r'D:\python\Deep learning\Fault diagnosis\data'
time_steps = 1024
window = 128
noise = True
snr = -10
feature= {}
for fault_type in FAULT_LABEL_DICT:
    sub_mat_data_array = []
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
            sub_mat_data = awgn(sub_mat_data, snr)
        sub_mat_data_array.append(sub_mat_data)
    feature[fault_type] = np.array(sub_mat_data_array)


# print(feature)
# print(feature['97'].shape) # (1898,1024)

for key in feature:
    plt.figure(figsize=(12,6))
    # plt.xlabel('time')
    plt.title(FAULT_DICT[key])
    plt.plot([x for x in range(1024)],feature[key][0,:],label = key)
    # plt.show()
    plt_dir = r'D:/python/Deep learning/Fault diagnosis/pyqt/picture/' + key + '.jpg'
    plt.savefig(plt_dir,
                dpi=300,  # 分辨率
                bbox_inches='tight',  # 去除多余空白
                transparent=True  # 透明背景
    )
