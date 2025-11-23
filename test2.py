import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
 
# 修正问题： 优化器的问题 第二个模型使用了第一个模型的优化器


# 分批次训练，一批 64 个
BATCH_SIZE = 64
# 所有样本训练 3 次
EPOCHS = 8
# 学习率设置为 0.0006
LEARN_RATE = 6e-4
 
# 若当前 Pytorch 版本以及电脑支持GPU，则使用 GPU 训练，否则使用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
# 训练集数据加载
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
# 构建训练集的数据装载器，一次迭代有 BATCH_SIZE 张图片
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
 
# 测试集数据加载
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)
# 构建测试集的数据加载器，一次迭代 1 张图片，我们一张一张的测试
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle = True)
 
"""
此处我们定义了一个 3 层的网络
隐藏层 1：40 个神经元
隐藏层 2：20 个神经元
输出层：10 个神经元
"""
class MLP(nn.Module):
    # 初始化方法
    # input_size 输入数据的维度
    # hidden_size 隐藏层的大小
    # num_classes 输出分类的数量
    def __init__(self,input_size,hidden_size,num_classes):
        # 调用父类的初始化方法
        super(MLP,self).__init__()
        # 定义第1个全连接层
        self.gate_proj = nn.Linear(input_size, hidden_size)
        # 定义ReLu激活函数
        # self.relu = nn.Silu()
        # 定义第2个全连接层
        self.up_proj = nn.Linear(input_size, hidden_size)
        # 定义第3个全连接层
        self.down_proj = nn.Linear(hidden_size, num_classes)
        
        self.act_fn = nn.functional.silu
    def forward(self,x):

        gate = self.gate_proj(x)
        gate = self.act_fn(gate)
        down_proj = self.down_proj( gate * self.up_proj(x))
        return down_proj

input_size = 28 * 28  #输入大小
hidden_size = 128  #隐藏层大小
num_classes = 2 #输出大小（类别数）
total_classes = 10
# 实例化DNN，并将模型放在 GPU 训练
model = MLP(input_size, hidden_size, num_classes).to(device)
# 同样，将损失函数放在 GPU
loss_fn = nn.MSELoss(reduction='mean').to(device)
loss_fn = nn.CrossEntropyLoss(reduction='mean')


# LEARN_RATE = 1e-3
#大数据常用Adam优化器，参数需要model的参数，以及学习率
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
 
dic = {0:0, 1:1, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1}


for epoch in range(EPOCHS):
    # 加载训练数据
    for step, data in enumerate(train_loader):
        x, y = data
        """
        因为此时的训练集即 x 大小为 （BATCH_SIZE, 1, 28, 28）
        因此这里需要一个形状转换为（BATCH_SIZE, 784）;
		
        y 中代表的是每张照片对应的数字，而我们输出的是 10 个神经元，
        即代表每个数字的概率
        因此这里将 y 也转换为该数字对应的 one-hot形式来表示
        """
        x = x.view(x.size(0), 784)
        yy = np.zeros((x.size(0), num_classes))
        for j in range(x.size(0)):
            
            yy[j][dic[y[j].item()]] = 1
        yy = torch.from_numpy(yy)
        yy = yy.float()
        x, yy = x.to(device), yy.to(device)
 
        # 调用模型预测
        output = model(x).to(device)
        # 计算损失值
        loss = loss_fn(output, yy)
        # 输出看一下损失变化
        print(f'EPOCH({epoch})   loss = {loss.item()}')
        # 每一次循环之前，将梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度下降，更新参数
        optimizer.step()
        
        # break
# 构造第二个网络
# for name, param in model.named_parameters():
#     param.requires_grad = False



hidden_size = 32 #隐藏层大小
next_num_classes = 5 #输出大小（类别数）
# 实例化DNN，并将模型放在 GPU 训练

input_size = 28 * 28  + 2
# input_size = 28 * 28  
next_model = MLP(input_size, hidden_size, total_classes).to(device)
# loss_fn = nn.CrossEntropyLoss(reduction='mean')


optimizer_next = torch.optim.Adam(next_model.parameters(), lr=LEARN_RATE)
for epoch in range(EPOCHS):
    # 加载训练数据
    for step, data in enumerate(train_loader):
        x, y = data
        """
        因为此时的训练集即 x 大小为 （BATCH_SIZE, 1, 28, 28）
        因此这里需要一个形状转换为（BATCH_SIZE, 784）;
		
        y 中代表的是每张照片对应的数字，而我们输出的是 10 个神经元，
        即代表每个数字的概率
        因此这里将 y 也转换为该数字对应的 one-hot形式来表示
        """
        x = x.view(x.size(0), 784)
        yy = np.zeros((x.size(0), total_classes))
        for j in range(x.size(0)):
            
            yy[j][[y[j].item() ] ] = 1
        yy = torch.from_numpy(yy)
        yy = yy.float()
        x, yy = x.to(device), yy.to(device)
 
        
        res = model(x).to(device)  


        # 得到 模型预测值
        tmp = (res).reshape(-1, 1).repeat(1, next_num_classes).reshape(-1, 10)
        # tmp.requires_grad = False
        # print(tmp.shape)
        # 调用模型预测
        # output = next_model(x).to(device)
        # 上一层的结果无法传递给这个网络
        
        x = torch.cat([x, res.detach()], dim=1)
        output = next_model(x)

        output =  output * tmp

        # print(output.shape)
        # print(yy.shape)
        # 计算损失值
        loss = loss_fn(output, yy)
        # 输出看一下损失变化
        print(f'EPOCH({epoch})   loss = {loss.item()}')

        # 每一次循环之前，将梯度清零
        optimizer_next.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度下降，更新参数
        optimizer_next.step()
        

        break

sum = 0
# test：
for i, data in enumerate(test_loader):
    
    x, y = data
    # 这里 仅对 x 进行处理
    x = x.view(x.size(0), 784)
    x, y = x.to(device), y.to(device)
    res = model(x).to(device)  
    # 得到 模型预测值
    tmp = (res).reshape(-1, 1).repeat(1, next_num_classes).reshape(-1, 10)

    x = torch.cat([x, res.detach()], dim=1)
    # 调用模型预测
    output = next_model(x).to(device)

    output =  output * tmp



    r = torch.argmax(output)
    r = torch.argmax(res)
    # 标签，即真实值
    l = y.item()
    l = dic[y.item()]
    sum += 1 if r == l else 0
    if i % 500 == 0:
        print(f'test({i})     DNN:{r} -- label:{l}')

    # if i > 2:
    #     break
 
print('accuracy：', sum / 10000)
