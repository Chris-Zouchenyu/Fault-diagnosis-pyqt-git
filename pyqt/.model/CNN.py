import torch
import torch.nn as nn

class CNNnet(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNnet, self).__init__()
        
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入通道1，输出通道32
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，输出尺寸减半
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入通道32，输出通道64
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，输出尺寸减半
        
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 全连接层，输入维度为64*8*8，输出维度为512
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)  # 输出层，输出维度为num_classes（10）
    
    def forward(self, x):
        # 将输入数据reshape为 (batch_size, 1, 32, 32)
        # 因为1024 = 1 * 32 * 32
        x = x.view(-1, 1, 32, 32)
        
        # 第一层卷积 + ReLU + 池化
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二层卷积 + ReLU + 池化
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 将特征图展平为一维向量
        x = x.view(x.size(0), -1)
        
        # 全连接层 + ReLU
        x = self.fc1(x)
        x = self.relu3(x)
        
        # 输出层
        x = self.fc2(x)
        
        return x

# 实例化模型
model = CNNnet(num_classes=10)

# 打印模型结构
# print(model)