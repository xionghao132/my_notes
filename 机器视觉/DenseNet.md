# DenseNet

## 代码

* 导入相关依赖

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

* 定义卷积块

```python
def conv_block(in_channels,out_channels):
    blk=nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
    )
    return blk
```

* 定义稠密块

```python
class DenseBlock(nn.Module):
    def __init__(self,num_conv,in_channels,out_channels):
        super(DenseBlock,self).__init__()
        net=[]
        for i in range(num_conv):
            in_c=in_channels+i*out_channels
            net.append(conv_block(in_c,out_channels))
        self.net=nn.ModuleList(net)
        self.out_channels=in_channels+num_conv*out_channels #计算输出通道数
    def forward(self,x):
        for blk in self.net:
            y=blk(x)
            x=torch.cat((x,y),dim=1) #合并通道维
        return x
```

* 过渡层

```python
#过渡层 通道数增加 得用1*1降低通道数
def transition_block(in_channels,out_channels):
    blk=nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels,out_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
    return blk
```

* 网络架构

```python
net=nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

num_channels,growth_rate=64,32         #num_channels为当前通道数 growth_rate为增长率->输出通道数
num_convs_in_dense_block=[4,4,4,4]     #卷积块数

for i,num_convs in enumerate(num_convs_in_dense_block):
    DB=DenseBlock(num_convs,num_channels,growth_rate)
    net.add_module(f'DengseBlock_{i}',DB)
    #上一个稠密块的输出通道
    num_channels=DB.out_channels
    #在稠密块之间加入通道数减半的过渡层
    if i !=len(num_convs_in_dense_block)-1:
        net.add_module(f'transition_block_{i}',transition_block(num_channels,num_channels//2))
        num_channels=num_channels//2
        
net.add_module('BN',nn.BatchNorm2d(num_channels))
net.add_module('relu',nn.ReLU())
net.add_module('global_avg_pool',nn.AdaptiveAvgPool2d((1,1)))
net.add_module('fc',nn.Sequential(nn.Flatten(),
                    nn.Linear(num_channels,10)
))
```

## 查看网络输出情况

```python
X=torch.rand(1,1,96,96)
for name,layer in net.named_children():
    X=layer(X)
    print(name,'output size',X.shape)
```

输出：

```python
0 output size torch.Size([1, 64, 48, 48])
1 output size torch.Size([1, 64, 48, 48])
2 output size torch.Size([1, 64, 48, 48])
3 output size torch.Size([1, 64, 24, 24])
DengseBlock_0 output size torch.Size([1, 192, 24, 24])
transition_block_0 output size torch.Size([1, 96, 12, 12])
DengseBlock_1 output size torch.Size([1, 224, 12, 12])
transition_block_1 output size torch.Size([1, 112, 6, 6])
DengseBlock_2 output size torch.Size([1, 240, 6, 6])
transition_block_2 output size torch.Size([1, 120, 3, 3])
DengseBlock_3 output size torch.Size([1, 248, 3, 3])
BN output size torch.Size([1, 248, 3, 3])
relu output size torch.Size([1, 248, 3, 3])
global_avg_pool output size torch.Size([1, 248, 1, 1])
fc output size torch.Size([1, 10])
```

