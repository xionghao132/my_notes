# Vgg

## 代码

* 导入相关依赖

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

* 定义vgg_block块

```python
def vgg_block(num_convs,in_channels,out_channels):
    blk=[]
    for i in range(num_convs):
        if i==0:
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)) #后面的参数可以保证图像大小不变
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2)) #这里会使宽和高减半
    return nn.Sequential(*blk)
```

* 训练参数

```python
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conv_arch=((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2,512,512)) #第一个表示卷积层的数量
#宽高减半5次 224/32=7
fc_features=512*7*7  #c*w*h
fc_hidden_units=4096
lr=0.001
epoch=5
```

* vgg11网络架构

```python
#实现vgg11
class Vgg11(nn.Module):
    def __init__(self):
        super(Vgg11,self).__init__()
        self.net=nn.Sequential()
        for i,(num_convs,in_channels,out_channels) in enumerate(conv_arch):
            #每经过一次vgg_block就减半
            self.net.add_module('vgg_block'+str(i+1),vgg_block(num_convs,in_channels,out_channels))
        #全连接层
        fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_features,fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units,fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units,10)
        )
        self.net.add_module('fc',fc)
    def forward(self,input):
        return self.net(input)
```

* 读取数据

```python
trans=Compose([
    Resize(224),
    ToTensor()
])
batch_size=128
train_data=FashionMNIST(root='./data',train=True,transform=trans,download=False) #下载完成记得修改为false
test_data=FashionMNIST(root='./data',train=False,transform=trans,download=False)
train_dataloaer=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)
```

* 实例化

```python
net=Vgg11()
optimizer=optim.Adam(net.parameters(),lr=lr)
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
net=Vgg11()
x=torch.rand(1,1,224,224)
for name,blk in next(net.children()).named_children():
    x=blk(x)
    print(name,'output shape: ',x.shape)
```

```python
vgg_block1 output shape:  torch.Size([1, 64, 112, 112])
vgg_block2 output shape:  torch.Size([1, 128, 56, 56])
vgg_block3 output shape:  torch.Size([1, 256, 28, 28])
vgg_block4 output shape:  torch.Size([1, 512, 14, 14])
vgg_block5 output shape:  torch.Size([1, 512, 7, 7])
fc output shape:  torch.Size([1, 10])
```

