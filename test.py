from models.linear_mpo import MPO, EmbeddingMPO, state_dict_matrix_to_mpo, LinearDecomMPO
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# 生成数据集
N = 100
D = 2
K = 2

torch.random.manual_seed(0)
X = torch.rand(N,D) # 输入特征
y = torch.zeros(N, dtype=torch.long) # 标签
y[:N//2] = 1
X[:N//2] = X[:N//2] - 0.7

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D, K)
        # self.linear = LinearDecomMPO(D, K, [2, 1, 1, 1, 1], [1, 1, 1, 1, 2])
        print(self.linear)

    def forward(self, x):
        return self.linear(x)


class MPOPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear = nn.Linear(D, K)
        self.linear = LinearDecomMPO(D, K, [2, 1, 1, 1, 1], [1, 1, 1, 1, 2])
        print(self.linear)

    def forward(self, x):
        return self.linear(x)


model = Perceptron()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Accuracy of the model: {model(X).argmax(1).eq(y).sum()/N}')

mpo_model = MPOPerceptron()
weight = model.linear.weight.detach()
bias = model.linear.bias.detach()
mpo_model.linear = LinearDecomMPO(D, K, [2, 1, 1, 1, 1], [1, 1, 1, 1, 2], _weight=weight, bias_tensor=bias)

optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'New Accuracy of the model: {model(X).argmax(1).eq(y).sum()/N}')


optimizer = optim.Adam(mpo_model.parameters(), lr=0.01)

for epoch in range(10):
    outputs = mpo_model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'New Accuracy of the mpo_model: {mpo_model(X).argmax(1).eq(y).sum()/N}')
