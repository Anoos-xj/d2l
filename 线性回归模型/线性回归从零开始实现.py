import torch
import random

"""生成数据集"""


def synthetic_data(w, b, num_examples):
	"""生成y=Xw+b+噪声"""
	X = torch.normal(0, 1, (num_examples, len(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
# print('features:',features[0],'label:',labels[0])

# plt.figure(figsize=(10,10))
# plt.scatter(features[:,1].detach().cpu().numpy(),labels.detach().cpu().numpy(),1)
# plt.show()

"""读取数据集"""


def data_iter(batch_size, features, labels):
	num_examples = len(features)
	indices = list(range(num_examples))
	"""这些样本是随机读取的，没有特定的顺序"""
	random.shuffle(indices)
	for i in range(0, num_examples, batch_size):
		batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
		yield features[batch_indices], labels[batch_indices]


batch_size = 10

# for X,y in data_iter(batch_size,features,labels):
# print(X,'\n',y)
# break

"""初始化模型参数"""

w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

"""定义模型"""


def linreg(X, w, b):
	"""线性回归模型"""
	return torch.matmul(X, w) + b


"""定义损失函数"""


def squared_loss(y_hat, y):
	"""均方损失"""
	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


"""定义优化算法"""


def sgd(params, lr, batch_size):
	"""小批量随机梯度下降"""
	with torch.no_grad():
		for p in params:
			if p.grad is not None:
				p -= lr * p.grad / batch_size
				p.grad.zero_()


"""训练"""
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
	for X, y in data_iter(batch_size, features, labels):
		l = loss(net(X, w, b), y)  # X和y的小批量损失
		# 因为l的形状是(batch_size,1)，而不是一个张量
		# l中的所有元素被加到一起，并以此计算关于[w,b]的梯度
		l.sum().backward()
		sgd([w, b], lr, batch_size)
	with torch.no_grad():
		train_l = loss(net(features, w, b), labels)
		print(f'epoch:{epoch + 1} ,loss:{float(train_l.mean()):f}')
