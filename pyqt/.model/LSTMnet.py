import torch.nn as nn
from torch.nn import Linear,ReLU,LSTM
from torch import ones

class LSTM_model(nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.lstm1 = LSTM(input_size = n_features, hidden_size = 200, num_layers = 1, batch_first = True)
        self.relu1 = ReLU()
        self.linear1 = Linear(in_features = 200, out_features = 1)
    def forward(self,x):
        x,(hn,cn) = self.lstm1(x)
        x = self.relu1(x)
        x = self.linear1(x)
        return x

# # 测试一下
# x = ones((30,1024,1))
# model = LSTM_model(1,10)
# y = model(x)
# print(x,y,x.shape,y.shape)