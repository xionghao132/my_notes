# LeNet

## 概述

LeNet是一个比较简单的卷积神经网络，是学习其他神经网络的基础，推荐自己对照网络架构图实现。它使用了两个卷积层与两个下采样，最后连接一个全连接层。

## 网络架构图

![](https://libraxiong-picture.oss-cn-hangzhou.aliyuncs.com/img/20220528220618.png)

从论文的图中，可以很容易复现这个网络。

==注意：==`@`前的数字表示通道数

参数计算：
$$
output\_size=\frac{input\_size+2×padding-kernel\_size}{stride}+1 \tag{1}
$$

## 代码

网络架构：

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.Linear(120,84),
            nn.Linear(84,10)
        )
    def forward(self,input):
        output = self.conv(input)
        return self.fc(output)
```

简单测试：

```python
net=LeNet()
x=torch.rand(1,1,32,32)
net(x).shape  #[1,10]
```

