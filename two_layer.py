import torch
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
 

import argparse
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 添加参数
parser.add_argument('--new', type=int, default=0)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--epoches', type=int, default=6)
parser.add_argument('--round', type=int, default=2)

# 解析参数
args = parser.parse_args()

# 分批次训练，一批 64 个
BATCH_SIZE = 64
# 所有样本训练 3 次
EPOCHS = args.epoches
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
    def __init__(self, input_size, intermediate_size, num_classes):
        # 调用父类的初始化方法
        super(MLP,self).__init__()
        # 定义第1个全连接层
        self.gate_proj = nn.Linear(input_size, intermediate_size)
        # 定义ReLu激活函数
        self.relu = nn.ReLU()
        # 定义第2个全连接层
        self.up_proj = nn.Linear(input_size, intermediate_size)
        # 定义第3个全连接层
        self.down_proj = nn.Linear(intermediate_size, input_size)
        
        self.act_fn = nn.functional.relu
        self.layer_norm = nn.LayerNorm(intermediate_size)

         

    def forward(self,x):

        gate = self.gate_proj(x)
        gate = self.act_fn(gate)

        x = gate
        # x = gate * self.up_proj(x)

        down_proj = self.down_proj(x)
        return down_proj


def rule(residual):
    res = residual / torch.max(residual, dim=1).values.unsqueeze(1)
    # print(res)
    ntopk = 2
    top_vals, top_inds = torch.topk(res, k=ntopk)

    ind = top_vals[:, ntopk - 1] > ( ( 1 / (ntopk) ) / 4)
    return ind


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, nlayers, new = 0, round = 0):
        super(MyModel, self).__init__()
        self.mlps = nn.ModuleList( [ MLP(input_size, hidden_size, num_classes) for _ in range(nlayers)] )

        self.out_proj = nn.ModuleList( [nn.Linear(input_size, num_classes) for _ in range(nlayers)] )

        self.input_layer_norm = nn.LayerNorm(input_size)
        self.layer_frozed = [False for _ in range(nlayers) ] 
        self.nlayers = nlayers
        self.input_size = input_size
        self.round = round

        self.new = new
        self.is_training = True


    def get_training_label(self, hidden_states):


        tensor = torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=device)
        for i in range(self.round):
            
            # [batch, 784]
            hidden_states = self._forward(hidden_states, i)

            # 找到所有满足rule的样本，然后去掉

            # print("find sample")
            output = self.out_proj[i](hidden_states)    
            
            any_stop = rule(output)

            # 返回所有需要early stop的 tensor
            tensor = tensor | any_stop

        # print(any_stop)
        return tensor
    

    def _forward(self, hidden_states, i):
        residual = hidden_states
        hidden_states  = self.input_layer_norm(residual)

        hidden_states = self.mlps[i](hidden_states)

        hidden_states = hidden_states + residual

        return hidden_states
    def forward(self, hidden_states):
        if self.round > 0 and self.new == 1:
            # print("第二轮 冻结第一轮的参数")
            
            for i in range(self.round):
                if not self.layer_frozed[i]:
                    model = self.mlps[i]
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                    model = self.out_proj[i]
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                self.layer_frozed[i] = True
        
        for i in range(self.nlayers):
            # x: [batch * 784]
            
            hidden_states = self._forward(hidden_states, i)

            # 如果正在训练，并且是训练round 0
            if self.new == 1:
                if self.is_training is True:
                    if self.round == i:
                        output = self.out_proj[i](hidden_states)
                        return output, i
                else:
                    output = self.out_proj[i](hidden_states)
                    ind = rule(output)
                    if ind.sum().item() == 0:
                        return (output, i)
                    
        output = self.out_proj[self.nlayers - 1](hidden_states)
        return output, self.nlayers - 1


def simple_recreate_trainloader(original_train_loader, 
        model, device, batch_size=BATCH_SIZE):
    all_new_x = []
    all_new_y = []
    from torch.utils.data import DataLoader, TensorDataset

    with torch.no_grad():
        for x, y in original_train_loader:
            # 您的处理逻辑
            x_flat = x.view(x.size(0), 784).to(device)
            y = y.to(device)
            ind = model.get_training_label(x_flat)
            
            # print(ind)
            # exit()
            if not ind.any():
                continue
            x_processed = x_flat[ind].reshape((-1, 1, 28, 28)).cpu()
            y_processed = y[ind].cpu()
            
            all_new_x.append(x_processed)
            all_new_y.append(y_processed)
    
    # 合并数据
    new_x = torch.cat(all_new_x, dim=0)
    new_y = torch.cat(all_new_y, dim=0)
    
    # 创建新数据集和loader
    new_dataset = TensorDataset(new_x, new_y)
    new_train_loader = DataLoader(
        new_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    print(f"创建完成! 新数据集: {len(new_dataset)} 个样本")
    return new_train_loader

input_size = 28 * 28  #输入大小
hidden_size = args.hidden #隐藏层大小
num_classes = 10 #输出大小（类别数）
# 实例化DNN，并将模型放在 GPU 训练

new = args.new
model = MyModel(input_size, hidden_size, num_classes, nlayers = args.round , new = new).to(device)
# 同样，将损失函数放在 GPU
loss_fn = nn.MSELoss(reduction='mean').to(device)
# loss_fn = nn.CrossEntropyLoss(reduction='mean')
 

    
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    

total_wrong = 0
total = 0
checked = False
total_round = args.round if args.new == 1 else 1
for round in range(0, total_round):




    model.round = round
    if round > 0:
        train_loader = simple_recreate_trainloader(train_loader, model, device = device)
              

    for epoch in range(EPOCHS):

        for step, data in enumerate(train_loader):
            x, y = data
            x = x.view(x.size(0), 784)
            yy = np.zeros((x.size(0), 10))
            for j in range(x.size(0)):
                yy[j][y[j].item()] = 1
            yy = torch.from_numpy(yy)
            yy = yy.float()
            x, yy = x.to(device), yy.to(device)


            output, layer = model(x)

            
            loss = loss_fn(output, yy)

            

            # 输出看一下损失变化
            if step % 100 == 0:
                print(f'EPOCH({epoch})  round = {round} loss = {loss.item()}')
            # 每一次循环之前，将梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度下降，更新参数
            optimizer.step()
        
 



model.is_training = False
sum = 0
# test：

total = 0
error = 0
sum_correct = 0
dic = { }

for i in range(args.round):
    dic[i] = { 'correct':0, 'total':0}
    
for i, data in enumerate(test_loader):
    
    x, y = data
    # 这里 仅对 x 进行处理
    x = x.view(x.size(0), 784)
    x, y = x.to(device), y.to(device)

    # if total_models == 1:
    res = model(x)

    if len(res) == 2:
        res, layer = res
        r = torch.argmax(res)
        l = y.item()
        dic[layer]['correct'] += 1 if r == l else 0
        dic[layer]['total'] += 1 



    total += 1
    # 得到 模型预测值
    r = torch.argmax(res)
    # 标签，即真实值
    l = y.item()
    sum += 1 if r == l else 0

    # print(f'test({i})     DNN:{r} -- label:{l}')
 
print("total = ", total)

print(dic)
print('accuracy：', sum / total)
