import torch.nn as nn
import torchvision.models as models
from torch import ones
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        
        # 加载预训练的ResNet18模型
        self.resnet18 = models.resnet18(pretrained=False)
        
        # 修改第一层卷积层，使其适应1024维的输入特征
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改最后的全连接层，使其输出10个类别
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
    
    def forward(self, x):
        # 将输入数据reshape为 (batch_size, 1, 32, 32)
        # 这里假设输入的x是(batch_size, 1024)，我们需要将其reshape为(batch_size, 1, 32, 32)
        # 因为1024 = 1 * 32 * 32
        x = x.view(-1, 1, 32, 32)
        
        # 通过ResNet18模型
        x = self.resnet18(x)
        
        return x

# # 实例化模型
# model = ResNet18(num_classes=10)
# y = model(ones(2,1024))
# # 打印模型结构
# print(model)
# print(y.shape)