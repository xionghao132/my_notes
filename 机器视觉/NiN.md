# NiN

## 代码

* 引入相关依赖

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor,Compose,Resize
import torch.optim as optim
```

* 定义nin块

```python
#定义nin块
def nin_block(in_channel,out_channel,kernel_size,stride,padding):
    nb=nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
        nn.ReLU(),
        nn.Conv2d(out_channel,out_channel,kernel_size=1), #1*1的卷积核
        nn.ReLU(),
        nn.Conv2d(out_channel,out_channel,kernel_size=1),
        nn.ReLU()
    )
    return nb
```

* 定义网络架构

```python
class NiN(nn.Module):
    def __init__(self):
        super(NiN,self).__init__()
        self.net=nn.Sequential(
            nin_block(1,96,11,4,0),
            nn.MaxPool2d(3,2),
            nin_block(96,256,5,1,2),
            nn.MaxPool2d(3,2),
            nin_block(256,384,3,1,1),
            nn.MaxPool2d(3,2),
            nn.Dropout(0.5),
            nin_block(384,10,3,1,1),
            nn.AdaptiveAvgPool2d((1,1)), #GAP层 [b,c,h,w]->[b,c,1,1]
            nn.Flatten()
        )
    def forward(self,input):
        return self.net(input)
```

* 训练参数

```python
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr=0.001
epoch=5

```

* 训练

```python
def train(net,train_dataloader,device,epoch):
    net=net.to(device)
    print('train on ',device)
    loss=nn.CrossEntropyLoss()
    batch_count=0
    for i in range(epoch):
        train_loss_sum,train_acc_sum,n=0.0,0.0,0
        for j,(input,target) in enumerate(train_dataloader):
            input=input.to(device)
            target=target.to(device)
            output=net(input)
            l=loss(output,target)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss_sum+=l.cpu().item()
            train_acc_sum+=(output.argmax(dim=1)==target).sum().cpu().item()
            n+=output.shape[0]  #加上batch_size
            batch_count+=1  #为了求损失的平均
        print(f'第{i+1}次epoch,train_loss_sum{train_loss_sum/batch_count},train_acc_sum{train_acc_sum/n}')
        
train(net,train_dataloader,device,epoch)
```

* 测试

```python
#注意device 得赋值
def test(net,test_dataloader,device):
    net.to(device)
    print('test on ',device)
    net.eval() #关闭drop out
    with torch.no_grad():
        acc_sum,n=0.0,0
        for j,(input,target) in enumerate(test_dataloader):
            input=input.to(device)
            target=target.to(device)
            output=net(input)
            acc_sum+=(output.argmax(dim=1)==target).float().sum()
            n+=output.shape[0]
        print(f'epoch,acc_sum{acc_sum/n}')

test(net,test_dataloader,device)
```

## 查看网络架构

```python
net=NiN()
x=torch.rand(1,1,224,224)
module=next(net.children()) #使用next是因为net.children()是ganator迭代对象并且只有一个self.net
for name,blk in module.named_children():
    x=blk(x)
    print(name,'output size',x.shape)
```

![](https://libraxiong-picture.oss-cn-hangzhou.aliyuncs.com/img/20220528221814.png)

