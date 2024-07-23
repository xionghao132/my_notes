# AlexNet

## 代码

* 导入相应的依赖

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
* 网络架构：

```python
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        #卷积层
        self.conv=nn.Sequential(
            nn.Conv2d(1,96,11,4),#in_channel,out_channel,kernel_size,stride
            nn.ReLU(),
            nn.MaxPool2d(3,2), #kernel_size stride
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            #之后不使用池化层减小输入的高和宽 之后卷积使用小窗口
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)     #输入[1,1,256,256] 输出[1,256,5,5]
        )
        #全连接层
        self.fc=nn.Sequential(
            nn.Linear(256*5*5,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )
    def forward(self,img):
        output=self.conv(img)
        return self.fc(output.view(img.shape[0],-1))
```

* 读入数据：

```python
trans=Compose([
    Resize(224),
    ToTensor()
])
batch_size=256
train_data=FashionMNIST(root='./data',train=True,transform=trans,download=False) #下载完成记得修改为false
test_data=FashionMNIST(root='./data',train=False,transform=trans,download=False)
train_dataloader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)
```

* 训练参数：

```python
#训练参数
lr=0.001
epoch=5
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=AlexNet()
optimizer=optim.Adam(net.parameters(),lr=lr)
```

* 训练：

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
            print(target.shape)
            print(output.shape)
            break
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

## 查看网络模型结构

```python
net=AlexNet()
x=torch.rand(1,1,224,224)
module=next(net.children()) #使用next是因为net.children()是ganator迭代对象并且只有一个self.net
for name,alx in module.named_children():
    x=alx(x)
    print(name,'output size',x.shape)
```

输出：

![](https://libraxiong-picture.oss-cn-hangzhou.aliyuncs.com/img/20220528222031.png)
