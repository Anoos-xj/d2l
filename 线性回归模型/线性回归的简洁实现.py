import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

"""生成数据集"""
true_w = torch.tensor([2, -3, 4], dtype=torch.float32)
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""读取数据集"""
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个Pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

"""定义模型"""
from torch import nn

# 修改模型以匹配特征数量
net = nn.Sequential(nn.Linear(3, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

"""定义损失函数"""
loss = nn.MSELoss()

"""定义优化算法"""
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

"""训练"""
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 确保y的形状是(batch_size, 1)
        l = loss(net(X), y.view(-1, 1))  # 使用view调整y的形状
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # 确保labels也是(batch_size, 1)形状
    l = loss(net(features), labels.view(-1, 1))
    print(f'epoch: {epoch+1}, loss: {l.item():f}')
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)