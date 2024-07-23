# Resnet

## 概述

`resnet`主要用于解决网络加深出现的退化现象，通过快捷连接的方式，很好地解决了深度神经网络难以训练的问题，可以说`resnet`撑起半边天，神经网络深度突破了$100$层,甚至可以突破$1000$层。

## 残差块

![](https://libraxiong-picture.oss-cn-hangzhou.aliyuncs.com/img/20220528220015.jpg)

## 网络结构图

![](https://libraxiong-picture.oss-cn-hangzhou.aliyuncs.com/img/20220528220238.png)

## 代码

* 代码

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

* ResNet残差块

```python
class Residual(nn.Module):
    def __init__(self,in_channel,out_channel,use_conv=False,stride=1):
        super(Residual, self).__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
        if use_conv:
            self.conv3=nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(out_channel)  #输入bn的通道数
        self.bn2=nn.BatchNorm2d(out_channel)
    def forward(self,x):
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        return F.relu(y+x)
```

* ResNet模块

```python
def resnet_block(in_chanel,out_channel,num_residuals,first_block=False):
    if first_block:
        assert in_chanel==out_channel #第一个模块的通道数通输入通道一致
    blk=[]
    for i in range(num_residuals):
        if i ==0 and not first_block:
            blk.append(Residual(in_chanel,out_channel,use_conv=True,stride=2))
        else:
            blk.append(Residual(out_channel,out_channel))
    return nn.Sequential(*blk)
```

* 构建网络

```python
net=nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)
net.add_module('resnet_block1',resnet_block(64,64,2,first_block=True))
net.add_module('resnet_block2',resnet_block(64,128,2))
net.add_module('resnet_block3',resnet_block(128,256,2))
net.add_module('resnet_block4',resnet_block(256,512,2))
net.add_module('global_avg_pool',nn.AdaptiveAvgPool2d((1,1)))
net.add_module('fc',nn.Sequential(nn.Flatten(),nn.Linear(512,10)))
```

* 测试网络

```python
x=torch.rand((1,1,224,224))
for name,layer in net.named_children():
    x=layer(x)
    print(name,'output',x.shape)
```

* 输出：

```python
0 output torch.Size([1, 64, 112, 112])
1 output torch.Size([1, 64, 112, 112])
2 output torch.Size([1, 64, 112, 112])
3 output torch.Size([1, 64, 56, 56])
resnet_block1 output torch.Size([1, 64, 56, 56])
resnet_block2 output torch.Size([1, 128, 28, 28])
resnet_block3 output torch.Size([1, 256, 14, 14])
resnet_block4 output torch.Size([1, 512, 7, 7])
global_avg_pool output torch.Size([1, 512, 1, 1])
fc output torch.Size([1, 10])
```

https://mbd.baidu.com/ma/s/rI5n0PO9

