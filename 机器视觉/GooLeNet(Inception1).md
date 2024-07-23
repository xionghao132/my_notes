# GooLeNet(Inception1)

## 概述

`GoogLeNet`是google推出的基于Inception模块的深度神经网络模型，在2014年的ImageNet竞赛中夺得了冠军，在随后的两年中一直在改进，形成了Inception V2、Inception V3、Inception V4等版本，本篇文章先介绍最早版本的GoogLeNet。

## Inception

![](https://libraxiong-picture.oss-cn-hangzhou.aliyuncs.com/img/20220528221013.png)

提升网络性能最直接的办法就是增加网络深度和宽度，这也就意味着巨量的参数。但是，巨量参数容易产生**过拟合**也会大大增加**计算量**。

使用5x5的卷积核仍然会带来巨大的计算量。借鉴Nin网络采用1x1卷积核来进行降维。

将全连接甚至一般的卷积都转化为稀疏连接，有助于提升网络性能。

## GooLeNet

整体架构：

![](https://libraxiong-picture.oss-cn-hangzhou.aliyuncs.com/img/20220528220939.jpg)

## 代码

* 导入依赖

```python
from typing import Callable
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor,Compose,Resize
import torch.optim as optim
from torch.nn import functional as F
```

* 构建`Inception`块

```python
class Inception(nn.Module):
    def __init__(self,in_channel,c1,c2,c3,c4):
        super(Inception, self).__init__()
        #线路1
        self.p1_1=nn.Conv2d(in_channel,c1,kernel_size=1)
        #线路2
        self.p2_1=nn.Conv2d(in_channel,c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        #线路3
        self.p3_1=nn.Conv2d(in_channel,c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        #线路4
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channel,c4,kernel_size=1)
    def forward(self,input):
        p1=F.relu(self.p1_1(input))
        p2=F.relu(self.p2_2(F.relu(self.p2_1(input))))
        p3=F.relu(self.p3_2(F.relu(self.p3_1(input))))
        p4=F.relu(self.p4_2(F.relu(self.p4_1(input))))
        print(p1.shape)
        print(p2.shape)
        print(p3.shape)
        print(p4.shape)
        return torch.cat((p1,p2,p3,p4),dim=1)    #[b,c,h,w] dim=1是通道维
```

* 构建模型

```
b1=nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b2=nn.Sequential(
    nn.Conv2d(64,64,kernel_size=1),
    nn.Conv2d(64,192,kernel_size=3,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b3=nn.Sequential(
    Inception(192,64,(96,128),(16,32),32),
    Inception(256,128,(128,192),(32,96),64),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b4=nn.Sequential(
    Inception(480,192,(96,208),(16,48),64),
    Inception(512,160,(112,224),(24,64),64),
    Inception(512,128,(128,256),(24,64),64),
    Inception(512,112,(144,288),(32,64),64),
    Inception(528,256,(160,320),(32,128),128),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
b5=nn.Sequential(
    Inception(832,256,(160,320),(32,128),128),
    Inception(832,384,(192,384),(48,128),128),
    nn.AdaptiveAvgPool2d((1,1))  #全局平均池化层
)
net=nn.Sequential(b1,b2,b3,b4,b5,
    nn.Flatten(),
    nn.Linear(1024,10)
)
```

## 查看网络每一层输出

```python
x=torch.rand(1,1,224,224)
for name,blk in net.named_children():
    x=blk(x)
    print(name,x.shape)
```

```python
#输出
0 torch.Size([1, 64, 56, 56])
1 torch.Size([1, 192, 28, 28])
2 torch.Size([1, 480, 14, 14])
3 torch.Size([1, 832, 7, 7])
4 torch.Size([1, 1024, 1, 1])
5 torch.Size([1, 1024])
6 torch.Size([1, 10])
```

[GoogLeNet](https://blog.csdn.net/shuzfan/article/details/50738394)