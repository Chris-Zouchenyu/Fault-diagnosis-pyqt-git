import torch.nn as nn
from torch.nn import Linear,ReLU,Dropout,BatchNorm1d,Sequential

class MLPnet(nn.Module):
    '''
    MLP神经网络
    '''
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Linear(1024,512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512,256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256,128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128,64),
            BatchNorm1d(64),
            ReLU(),
            Dropout(0.5),
            Linear(64,10),
        )
    def forward(self,x):
        x = self.model1(x)
        return x

