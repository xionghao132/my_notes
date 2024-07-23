# CGAN

```python
#%%

import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch import optim
import os
import numpy as np


#%%

# 设置超参数
batch_size = 100
learning_rate = 0.0002
epochsize = 90
sample_dir = "images3"

# 创建生成图像的目录
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

#%%

# 生成器结构
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(

            nn.Linear(110, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )
 #noise:z [100,100] label:[100]
    def forward(self, noise, label):
        #self.label_emb():[100,10] out [100,110]
        out = torch.cat((noise, self.label_emb(label)), -1)  #dim=-1表示最后一维
        img = self.model(out)     # torch.Size([100, 784])
        img = img.view(img.size(0), 1, 28, 28)     # torch.Size([100, 1, 28, 28])
        return img

#%%

# 鉴别器结构
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(

            nn.Linear(794, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        img = img.view(img.size(0), -1)     # torch.Size([100, 784])
        #self.label_emb(label) [100,10]
        x = torch.cat((img, self.label_emb(label)), -1)     # torch.Size([100, 794])
        x = self.model(x)   # torch.Size([100, 1])
        return x

#%%

# 训练集下载
mnist_traindata = datasets.MNIST('./data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
]), download=False)
mnist_train = DataLoader(mnist_traindata, batch_size=batch_size, shuffle=True, pin_memory=True)

#%%

G = Generator()
D = Discriminator()

#%%

# 导入之前的训练模型
G.load_state_dict(torch.load('G_plus.ckpt'))
D.load_state_dict(torch.load('D_plus.ckpt'))

#%%

# 设置优化器与损失函数,二分类的时候使用BCELoss较好,BCEWithLogitsLoss是自带一层Sigmoid
# criteon = nn.BCEWithLogitsLoss()
criteon = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate)


#%%

# 开始训练
print("start training")
for epoch in range(epochsize):

    D_loss_total = 0
    G_loss_total = 0
    total_num = 0

    for batchidx, (realimage, realimage_label) in enumerate(mnist_train):
        # realimage = realimage.to(device)
        D_optimizer.zero_grad()
        realscore = torch.ones(realimage.size(0), 1)   # value：1 torch.Size([100, 1])
        fakescore = torch.zeros(realimage.size(0), 1)   # value：0 torch.Size([100, 1])
        # 随机sample出噪声与标签，生成假图像
        z = torch.randn(realimage.size(0), 100)  # [100,100]   有100个 100长度的向量
        fakeimage_label = torch.LongTensor(np.random.randint(0, 10, realimage.size(0)))
        d_realimage_loss = criteon(D(realimage, realimage_label), realscore)
        
        fakeimage = G(z, fakeimage_label)  #fakeimage [100,1,28,28]
        d_fakeimage_loss = criteon(D(fakeimage.detach(), fakeimage_label), fakescore)  #detach此时不能传递梯度给生成图像 不然就需要重新生成
        D_loss = d_realimage_loss + d_fakeimage_loss
        
        # 参数训练三个步骤
        D_loss.backward()
        D_optimizer.step()
        # 计算一次epoch的总损失
        D_loss_total += D_loss
        
        G_optimizer.zero_grad()
        G_loss = criteon(D(fakeimage, fakeimage_label), realscore)
        # 参数训练三个步骤
        G_loss.backward()
        G_optimizer.step()

        # 计算一次epoch的总损失
        G_loss_total += G_loss

        # 打印相关的loss值
        if batchidx % 200 == 0:
            print("batchidx:{}/{}, D_loss:{}, G_loss:{}".format(batchidx, len(mnist_train), D_loss, G_loss))

    # 打印一次训练的loss值
    print('Epoch:{}/{}, D_loss:{}, G_loss:{}'.format(epoch, epochsize, D_loss_total / len(mnist_train),
                                                                   G_loss_total / len(mnist_train)))

    # 保存生成图像
    z = torch.randn(batch_size, 100)
    label = torch.LongTensor(np.array([num for _ in range(10) for num in range(10)]))
    save_image(G(z, label).data, os.path.join(sample_dir, 'images-{}.png'.format(epoch + 61)), nrow=10, normalize=True)

    # 保存网络结构
    torch.save(G.state_dict(), 'G_plus.ckpt')
    torch.save(D.state_dict(), 'D_plus.ckpt')


#%%

#测试代码
import torch
from torch import nn
from torchvision.utils import save_image
import os
import numpy as np



#%%

# 设置超参数
batch_size = 100
# learning_rate = 0.0002
# epochsize = 80
sample_dir = "test_images"

#%%

# 创建生成图像的目录
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

#%%

# 导入训练好的模型
G = Generator()
G.load_state_dict(torch.load('G_plus.ckpt'))

# 保存图像
z = torch.randn(batch_size, 100)
# label = torch.LongTensor(np.array([num for _ in range(10) for num in range(10)]))
label = torch.tensor([7,8,1,3,4,2,6,5,9,0]*10)
# label = torch.full([100], 9)

# label = []
# for i in range(10):
#     for j in range(10):
#         label.append(i)
#
# label = torch.tensor(label)
print(label)
print("label.shape:", label.size())

save_image(G(z, label).data, os.path.join(sample_dir, 'images.png'), nrow=10, normalize=True)
```









[CSDN编程社区 (smartapps.cn)](https://yebd1h.smartapps.cn/pages/blog/index?_swebFromHost=baiduboxapp&blogId=117451095&_swebfr=1)