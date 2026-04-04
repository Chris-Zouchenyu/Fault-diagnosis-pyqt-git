# Fault-diagnosis-pyqt-git
A fault diagnosis system for bearing equipment based on PyQt5 and deep learning, integrating model training, data processing, and visual diagnosis.  
（基于 PyQt5 和深度学习的轴承设备故障诊断系统，融合模型训练、数据处理、可视化诊断能力） 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)  

Recently, I learned a pyqt5 plugin, which is a very suitable plugin for GUI development. This is a fault diagnosis system that I developed using pyqt.  
(最近，我学习了一个名为pyqt5的插件，它非常适合用于图形用户界面（GUI）开发。这是我使用pyqt开发的一个故障诊断系统。) 

HOW TO LEARN PYTQ+ERIC6:  
https://www.bilibili.com/video/BV1L54y157P1?spm_id_from=333.788.videopod.episodes&vd_source=227625ed7ba6b0cfb5a1330e886cac76&p=27

I have beautified the README. The source code is actually quite simple.  
(README我进行了美化处理 源代码其实很简单)  
Created by Zouchenyu from Xi`an Jiaotong university.  
（由西安交通大学研究生邹晨宇 开发）

## 📋 Overview
该仓库是一个基于 **PyQt5** 开发的故障诊断系统，聚焦于轴承类设备故障诊断任务，融合深度学习模型训练、GUI 交互、数据处理等核心能力，是集**数据加载、模型训练、故障诊断可视化**于一体的完整桌面应用。

## 🎯 Core Features（核心功能）
### 1. 登录验证
- 基于 Qt Designer 设计的登录界面，校验固定账号（Chris/123456）后进入主界面；
- 支持窗口尺寸限定、密码隐藏输入等交互细节，界面样式轻量化设计。
![image](https://github.com/Chris-Zouchenyu/Fault-diagnosis-pyqt-git/blob/main/pyqt/.others/4.png)  
### 2. 深度学习模型训练
- **多模型支持**：实现 MLP/CNN/LSTM/ResNet18 四种模型，适配 1024 维故障时序数据；
- **数据处理**：
  - 加载 `.mat` 格式轴承驱动端振动数据，按 1024 步长+128 滑动窗口切割样本；
  - 支持添加高斯白噪声（AWGN）增强鲁棒性，自定义信噪比（SNR）；
  - 内置数据标准化、训练/验证/测试集划分（自定义比例）；
- **灵活配置**：支持自定义学习率、batchsize、epoch、模型类型等训练参数；
- **多线程训练**：基于 PyQt5 `QThread` 封装训练流程，避免 GUI 界面卡顿。
![image](https://github.com/Chris-Zouchenyu/Fault-diagnosis-pyqt-git/blob/main/pyqt/.others/1.png)  
### 3. 实时故障诊断与可视化
- **数据可视化**：绘制故障时序数据曲线（支持中文显示），支持图片保存；
- **故障识别**：加载训练好的模型，预测 10 类故障/正常状态；
- **结果展示**：支持 Excel 数据导入、预测结果打印、时序数据可视化。 
![image](https://github.com/Chris-Zouchenyu/Fault-diagnosis-pyqt-git/blob/main/pyqt/.others/3.png)  
## 🔧 Tech Stack（技术栈）
### Frontend/GUI
- 核心框架：PyQt5（Qt Designer、信号槽、QThread 多线程）；
- 界面组件：登录窗口、标签页布局、各类交互控件、图片显示；
- 可视化：Matplotlib（自定义字体/分辨率、图片保存）。

### Backend/Algorithm
- 数据处理：NumPy、Pandas、Scipy（.mat 加载/噪声生成）、Scikit-learn；
- 深度学习：PyTorch（模型构建/训练/保存）、torchkeras；
- 数据适配：1024 维时序数据 → 32×32 二维格式（CNN/ResNet）/序列格式（LSTM）。

## 📊 Data & Fault Types（数据与故障类型）
### Data Source
- 数据集：`.mat` 格式轴承驱动端（_DE_time）振动数据；
- 预处理：1024 时间步切割样本，支持 -10dB 高斯白噪声增强。

### Fault Label System
涵盖 1 类正常 + 9 类故障（按故障位置/尺寸划分）：

| 标签值 | 故障类型                |
|--------|-------------------------|
| 0      | 正常数据                |
| 1      | 7密耳 内圈故障          |
| 2      | 7密耳 滚动体故障        |
| 3      | 7密耳 外圈故障          |
| 4      | 14密耳 内圈故障         |
| 5      | 14密耳 滚动体故障       |
| 6      | 14密耳 外圈故障         |
| 7      | 21密耳 内圈故障         |
| 8      | 21密耳 滚动体故障       |
| 9      | 21密耳 外圈故障         |

## 📁 Project Structure（项目结构）
```
Fault-diagnosis-pyqt-git/
├── README.md               # 仓库说明、功能介绍、学习参考链接
├── pyqt/
│   ├── .mainwindow/        # 主窗口核心逻辑
│   │   ├── index.py        # 程序入口（登录窗口 + 主窗口调度）
│   │   ├── window1.py      # 登录窗口业务逻辑
│   │   ├── window2.py      # 主功能窗口（数据加载、模型训练、诊断）
│   │   ├── Ui_window1.py   # 登录窗口UI自动生成代码
│   │   ├── Ui_window2.py   # 主窗口UI自动生成代码
│   ├── .model/             # 深度学习模型定义
│   │   ├── MLP.py          # MLP模型
│   │   ├── CNN.py          # CNN模型
│   │   ├── LSTMnet.py      # LSTM模型
│   │   ├── ResNet.py       # ResNet18模型（适配1024维输入）
│   ├── .others/            # 辅助脚本
│   │   ├── plot.py         # 数据可视化、图片保存
│   │   ├── test_3.5.py     # 模型预测、结果测试
│   ├── .picture/           # 可视化图片存储目录
│   ├── .eric7project/      # Eric6（PyQt开发工具）项目配置
│   ├── .jedi/              # 代码补全/分析工具配置
```
## ✨ Features & Scenarios（特色与场景）
- 学习价值：适合入门 PyQt 桌面开发 + 深度学习故障诊断；
- 工程化：多线程防卡顿、模型与业务解耦、数据流程标准化；
- 可扩展：支持优化模型精度、扩展故障类型、兼容更多数据格式（如 CSV）。

## 🚀 Quick Start（快速开始）
### Install Dependencies
```bash
pip install PyQt5 PyQt5-tools numpy pandas scipy scikit-learn torch torchkeras matplotlib
```
## 📄 License
This project is for learning and research purposes only.  
Created by Zouchenyu from Xi`an Jiaotong university
