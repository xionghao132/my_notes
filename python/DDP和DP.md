# DDP和DP

## 概述

DistributedDataParallel（DDP）是一个支持多机多卡、分布式训练的深度学习工程方法。PyTorch现已原生支持DDP，可以直接通过torch.distributed使用，超方便，不再需要难以安装的apex库啦！

apex也可以了解一下



DDP比DP快，DDP启动多个进程，不会受到python GIL的限制，

一般来说，DDP都是显著地比DP快，能达到略低于卡数的加速比（例如，四卡下加速3倍）。所以，其是目前最流行的多机多卡训练方法。



在分类上，DDP属于Data Parallel。简单来讲，就是通过提高batch size来增加并行度。

依赖：PyTorch(gpu)>=1.5，python>=3.6



DP使用的是多线程，DDP使用的是多线程。

## 安装

DDP需要安装nccl,需要与系统和CUDA版本进行匹配。



有一个nccl-tests的github库可以下载测试

正常都可以通过root权限，直接安装nccl，但是也有很多时候是无root权限的，

由于nccl搭建好了，还是不行，应该就是版本问题，所以最好的办法就是重新建立一个新环境

这里服务器CUDA驱动是11.0，我下载的驱动是10.2’



* 从github上将NCCL的仓库拉到本地：

```sh
git clone https://github.com/NVIDIA/nccl.git   #不建议直接下载，应该看电脑和CUDA对应版本，然后从tag中找对应版本下载
```

* 编译

```sh
cd nccl
make -j12 src.build BUILDDIR=/data3/xhao/work_tools/software/nccl CUDA_HOME=/usr/local/cuda 
#-j12表示使用12个核心 BUILDDIR表示编译后的存储路径  CUDA_HOME表示CUDA目录，默认是/usr/local/cuda
```

* 添加环境变量

```sh
#.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenz/software/nccl/lib
export PATH=$PATH:/data3/xhao/work_tools/software/nccl/bin  #冒号是将多个变量隔开

#这个地方要检查编译出来的文件，如果没有bin目录，就写下面的路径
export PATH=$PATH:/data3/xhao/work_tools/software/nccl/include

source ~/.bashrc
```



```sh
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html



[NCCL无root权限编译安装_nccl文件夹下没有bin_是暮涯啊的博客-CSDN博客](https://blog.csdn.net/longshaonihaoa/article/details/121794715)

[(3 封私信 / 80 条消息) SuninKingdom - 知乎 (zhihu.com)](https://www.zhihu.com/people/suninkingdom)



## 区别

- **CUDA**：为“GPU通用计算”构建的运算平台。
- **cudnn**：为深度学习计算设计的软件库，这个地方的nn可以直接理解成神经网络。
- **CUDA Toolkit (nvidia)**： CUDA完整的工具安装包，其中提供了 Nvidia 驱动程序、开发 CUDA 程序相关的开发工具包等可供安装的选项。包括 CUDA 程序的编译器、IDE、调试器等，CUDA 程序所对应的各式库文件以及它们的头文件。
- **CUDA Toolkit (Pytorch)**： CUDA不完整的工具安装包，其主要包含在使用 CUDA 相关的功能时所依赖的动态链接库。不会安装驱动程序。
- **NVCC** ：CUDA的编译器，只是 CUDA Toolkit 中的一部分



通过conda安装的CUDA Toolkit包含会使用到的so文件,通过 Anaconda 安装的应用程序包位于安装目录下的 /pkg 文件夹中

显示CUDA版本

```sh
nvidia-smi
nvcc -V
```

CUDA主要有两个API：runtime API、driver API

用于支持driver API的必要文件(如libcuda.so)是由GPU driver installer安装的。
用于支持runtime API的必要文件(如libcudart.so以及nvcc)是由CUDA Toolkit installer安装的。
nvidia-smi属于driver API、nvcc属于runtime API。
nvcc属于CUDA compiler-driver tool，只知道runtime API版本，甚至不知道是否安装了GPU driver。





查找CUDA位置

```
echo $CUDA_HOME
```



Pytorch 会首先定位一个 cuda 安装目录( 来获取所需的特定版本 cuda 提供的可执行程序、库文件和头文件等文件 )。具体而言，Pytorch 首先尝试获取环境变量 CUDA_HOME/CUDA_PATH 的值作为运行时使用的 cuda 目录。若直接设置了 CUDA_HOME/CUDA_PATH 变量，则 Pytorch 使用 CUDA_HOME/CUDA_PATH 指定的路径作为运行时使用的 cuda 版本的目录。

在确定好使用的 cuda 路径后，基于 cuda 的 Pytorch 拓展即会使用确定好的 cuda 目录中的可执行文件( /bin )、头文件( /include )和库文件( /lib64 )完成所需的编译过程。



```sh
export CUDA_HOME=/home/test/cuda-10.1/        　　　#设置全局变量 CUDA_HOME
export PATH=$PATH:/home/test/cuda-10.1/bin/        #在 PATH 变量中加入需要使用的 cuda 版本的路径,使得系统可以使用 cuda 提供的可执行文件，包括 nvcc
```

[【精选】一文讲清楚CUDA、CUDA toolkit、CUDNN、NVCC关系_健0000的博客-CSDN博客](https://blog.csdn.net/qq_41094058/article/details/116207333)

## DP

在DP模式中，总共只有一个进程（受到GIL很强限制）。master节点相当于参数服务器，其会向其他卡广播其参数；在梯度反向传播后，各卡将梯度集中到master节点，master节点对搜集来的参数进行平均后更新参数，再将参数统一发送到其他卡上。这种参数更新方式，会导致master节点的计算任务、通讯量很重，从而导致网络阻塞，降低训练速度。

但是DP也有优点，优点就是代码实现简单。



直接使用`DataParallel`包裹就行,

```python
#注意model得先放在cuda上，model不是并行的，数据是并行的，所以这里是扩大batch_size
#网上说model会在这些device都进行扩展，所以这样加速的
net=torch.nn.DataParallel(model.cuda(),device_ids=[0,1,2])  
if hasattr(net,'module'):
	net=net.module
```



## DDP

在 pytorch 1.0 之后，官方终于对分布式的常用方法进行了封装，支持 all-reduce，broadcast，send 和 receive 等等。通过 MPI 实现 CPU 通信，通过 NCCL 实现 GPU 通信。官方也曾经提到用 DistributedDataParallel 解决 DataParallel 速度慢，GPU 负载不均衡的问题，目前已经很成熟了～



### 概念

* **group**

即进程组。默认情况下，只有一个组。这个可以先不管，一直用默认的就行。

* **world size**

表示全局的并行数，简单来讲，就是2x8=16。

```python3
# 获取world size，在不同进程里都是一样的，得到16
torch.distributed.get_world_size()
```

* **rank**

表现当前进程的序号，用于进程间通讯。对于16的world sizel来说，就是0,1,2,…,15。
注意：rank=0的进程就是master进程。

```text
# 获取rank，每个进程都有自己的序号，各不相同
torch.distributed.get_rank()
```

* **local_rank**

又一个序号。这是每台机子上的进程的序号。机器一上有0,1,2,3,4,5,6,7，机器二上也有0,1,2,3,4,5,6,7

```python
# 获取local_rank。一般情况下，你需要用这个local_rank来手动设置当前模型是跑在当前机器的哪块GPU上面的。
torch.distributed.local_rank()
```





DDP有不同的使用模式。**DDP的官方最佳实践是，每一张卡对应一个单独的GPU模型（也就是一个进程），在下面介绍中，都会默认遵循这个pattern。**
举个例子：我有两台机子，每台8张显卡，那就是2x8=16个进程，并行数是16。

但是，我们也是可以给每个进程分配多张卡的。总的来说，分为以下三种情况：

1. 每个进程一张卡。这是DDP的最佳使用方法。
2. 每个进程多张卡，复制模式。一个模型复制在不同卡上面，每个进程都实质等同于DP模式。这样做是能跑得通的，但是，速度不如上一种方法，一般不采用。
3. 每个进程多张卡，并行模式。一个模型的不同部分分布在不同的卡上面。例如，网络的前半部分在0号卡上，后半部分在1号卡上。这种场景，一般是因为我们的模型非常大，大到一张卡都塞不下batch size = 1的一个模型



### 原理

假如我们有N张显卡，

1. （缓解GIL限制）在DDP模式下，会有N个进程被启动，每个进程在一张卡上加载一个模型，这些模型的参数在数值上是相同的。
2. （Ring-Reduce加速）在模型训练时，各个进程通过一种叫Ring-Reduce的方法与其他进程通讯，交换各自的梯度，从而获得所有进程的梯度；
3. （实际上就是Data Parallelism）各个进程用平均后的梯度更新自己的参数，因为各个进程的初始参数、更新梯度是一致的，所以更新后的参数也是完全相同的。



### Python GIL

Python GIL的存在使得，一个python进程只能利用一个CPU核心，不适合用于计算密集型的任务。
使用多进程，才能有效率利用多核的计算资源。

而DDP启动多进程训练，一定程度地突破了这个限制。



### Ring-Reduce梯度合并

传统的梯度合并方法`tree allreduce`



![v2-91128397dc6575d5e2751fbe012a0023_r](images/v2-91128397dc6575d5e2751fbe012a0023_r.png)

`GPU1~4`卡负责网络参数的训练，每个卡上都布置了相同的深度学习网络，每个卡都分配到不同的数据的`minibatch`。每张卡训练结束后将网络参数同步到`GPU0`，也就是`Reducer`这张卡上，然后再求参数变换的平均下发到每张计算卡。

1. **问题一**，每一轮的训练迭代都需要所有卡都将数据同步完做一次Reduce才算结束。如果卡数比较少的情况下，其实影响不大，但是如果并行的卡很多的时候，就涉及到计算快的卡需要去等待计算慢的卡的情况，造成计算资源的浪费。
2. **问题二**，每次迭代所有的计算GPU卡多需要针对全部的模型参数跟Reduce卡进行通信，如果参数的数据量大的时候，那么这种通信开销也是非常庞大，而且这种开销会随着卡数的增加而线性增长。



改进的通信算法Ring-Reduce

![v2-d725a864c17515b0677e37877693d84b_720w](images/v2-d725a864c17515b0677e37877693d84b_720w.png)

Ring-Reduce是一种分布式程序的通讯方法。

- 因为提高通讯效率，Ring-Reduce比DP的parameter server快。

- - 其避免了master阶段的通讯阻塞现象，n个进程的耗时是o(n)。

简单来说所有的worker卡将梯度发送给main卡进行求和取平均，但这种方式一个很大的问题就是随着机器GPU卡数的增加，main卡的通信量也是线性增长。在通信带宽确定的情况下（不考虑延迟），GPU卡数越多，通信量越大，通信时间越长， 所需要的时间会随着GPU数量增长而线性增长。

第一步是scatter-reduce, 然后是allgather。scatter-reduce操作将GPU交换数据，是的每个GPU可得到最终结果的一个块。在allgather中，GPU将交换这些块，使得所有GPU得到完整的结果。

整个通信过程中，每个GPU的通信量不再随着GPU增加而增加，通信的速度受到环中相邻GPU之间最慢的链接（最低的带宽）的限制（不考虑延迟）。



因为每张卡上面的网络结构是固定的，所以里面的参数结构相同。每次通信的过程中，只将参数send到右手边的卡，然后从左手边的卡receive数据。经过不断地迭代，就会实现整个参数的同步，也就是reduce。

### DistributedSampler机制

1. 第一个就是 DistributedSampler 会把数据划分成 num_gpu 份，不同的 GPU 拿自己那一份，那么是不是训练过程中每个 GPU 只能见到自己那一份数据，不能看到别的 GPU 的数据，这样是否每个 GPU 会过拟合自己那份数据？（评论区有人回复了，答案应该是不会，因为GPU之间要同步梯度和参数）
2. 第二个就是我在测试的时候发现，尽管 DistributedSampler 默认是 shuffle=True，但是每个epoch和每次运行（重新运行整个程序），epoch之间每个 GPU 的输出顺序是一样的，没有 shuffle 成功。其次每次运行的输出顺序都是一样的，这就非常奇怪了。



## 代码

### 基本使用

单GPU代码

```python3
## main.py文件
import torch

# 构造模型
model = nn.Linear(10, 10).to(local_rank)

# 前向传播
outputs = model(torch.randn(20, 10).to(rank))
labels = torch.randn(20, 10).to(rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
# 后向传播
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()

## Bash运行
python main.py
```

加入DDP的代码

```python
## main.py文件
import torch
# 新增：
import torch.distributed as dist

# 新增：从外面得到local_rank参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# 新增：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
# 新增：构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 前向传播
outputs = model(torch.randn(20, 10).to(rank))
labels = torch.randn(20, 10).to(rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
# 后向传播
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()


## Bash运行
# 改变：使用torch.distributed.launch启动DDP模式，
#   其会给main.py一个local_rank的参数。这就是之前需要"新增:从外面得到local_rank参数"的原因
python -m torch.distributed.launch --nproc_per_node 4 main.py
```

torch==1.8基本不会出错，torch>=1.9就不稳定

WARNING:torch.distributed.elastic.agent.server.api:Received 1 death signal, shutting down workers

```sh
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```



**注意**：多进程虽然运行的是同一份代码，但是要处理好不同进程之间的关系，



```python
## main.py文件
import torch
import argparse

# 新增1:依赖
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 新增2：从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数，后面还会介绍。所以不用考虑太多，照着抄就是了。
#       argparse是python的一个系统库，用来处理命令行调用，如果不熟悉，可以稍微百度一下，很简单！
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# 新增3：DDP backend初始化
#   a.根据local_rank来设定当前使用哪块GPU
torch.cuda.set_device(local_rank)
#   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
dist.init_process_group(backend='nccl')

# 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
#       如果要加载模型，也必须在这里做哦。
device = torch.device("cuda", local_rank)
model = nn.Linear(10, 10).to(device)
# 可能的load模型...

# 新增5：之后才是初始化DDP模型
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

### **前向与后向传播**

有一个很重要的概念，就是数据的并行化。
我们知道，DDP同时起了很多个进程，但是他们用的是同一份数据，那么就会有数据上的冗余性。也就是说，你平时一个epoch如果是一万份数据，现在就要变成1*16=16万份数据了。
那么，我们需要使用一个特殊的sampler，来使得各个进程上的数据各不相同，进而让一个epoch还是1万份数据。



```python
my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True)
# 新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。
#       sampler的原理，后面也会介绍。
train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
# 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=train_sampler)

for epoch in range(num_epochs):
    # 新增2：设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        prediction = model(data)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        optimizer.step()
```



### 汇总数据

* `torch.distributed.all_gather`

训练BPR中的代码

```python
query_repr_list = [torch.empty_like(query_repr) for _ in range(dist.get_world_size())]
dist.all_gather(query_repr_list, query_repr)   #将各自节点的数据传入
query_repr = torch.cat(query_repr_list, dim=0)
```

* `torch.distributed.all_reduce`

训练DPR中的代码

`dist.all_reduce(tensor, op=ReduceOp.SUM)`这里还有一个op参数，一般都是默认相加

```pytorch
import torch.distributed as dist
if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)
```



### 单GPU运行

我们会遇到一些情况，并不需要所有的GPU都运行这个内容，我们一般都默认在`local_rank=0`上执行这些内容。

比如我们遇到发现没有数据集的时候，只需要通过一个进程去下载这个数据集，其他进行等待即可，后面的模型保存加个`if`判断即可，

```python
from contextlib import contextmanager 
 
@contextmanager #@contextmanager装饰器定义的上下文管理器函数
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    # 这里的用法其实就是协程的一种哦。
    yield     #使用yield语句将控制权交给上下文管理器的使用者。
    if rank == 0:
        torch.distributed.barrier()
        
with torch_distributed_zero_first(rank):
    if not check_if_dataset_exist():
        download_dataset()
    load_dataset()     
```



### 随机种子

```python

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

rank = torch.distributed.get_rank()
init_seeds(1 + rank)        
```

[torch.distributed多卡/多GPU/分布式DPP(二)—torch.distributed.all_reduce(reduce_mean)barrier控制进程执行顺序&seed随机种子_dist.all_reduce_hxxjxw的博客-CSDN博客](https://blog.csdn.net/hxxjxw/article/details/126238957)

### 其他

- 保存参数

```python
# 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
#    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
# 2. 我只需要在进程0上保存一次就行了，避免多次保存重复的东西。
if dist.get_rank() == 0:
    torch.save(model.module, "saved_model.ckpt")
```



- 理论上，在没有buffer参数（如BN）的情况下，DDP性能和单卡Gradient Accumulation性能是完全一致的。

- - 并行度为8的DDP 等于 Gradient Accumulation Step为8的单卡

  - 速度上，DDP当然比Graident Accumulation的单卡快；

  - - 但是还有加速空间。请见DDP系列第三篇：实战。

  - 如果要对齐性能，需要确保喂进去的数据，在DDP下和在单卡Gradient Accumulation下是一致的。

  - - 这个说起来简单，但对于复杂模型，可能是相当困难的。

### 调用方式

像我们在QuickStart里面看到的，DDP模型下，python源代码的调用方式和原来的不一样了。现在，需要用`torch.distributed.launch`来启动训练。

- 作用

- - 在这里，我们给出分布式训练的**重要参数**：

  - - **有多少台机器？**

    - - **--nnodes**

    - **当前是哪台机器？**

    - - **--node_rank**

    - **每台机器有多少个进程？**

    - - **--nproc_per_node**

    - 高级参数（可以先不看，多机模式才会用到）

    - - 通讯的address
      - 通讯的port

- 实现方式

- - 我们需要在每一台机子（总共m台）上都运行一次`torch.distributed.launch`

  - 每个`torch.distributed.launch`会启动n个进程，并给每个进程一个`--local_rank=i`的参数

  - - 这就是之前需要"新增:从外面得到local_rank参数"的原因

  - 这样我们就得到n*m个进程，world_size=n*m

**单机模式**

```bash
## Bash运行
# 假设我们只在一台机器上运行，可用卡数是8
python -m torch.distributed.launch --nproc_per_node 8 main.py
```

**多机模式**

复习一下，master进程就是rank=0的进程。
在使用多机模式前，需要介绍两个参数：

- 通讯的address

- - `--master_address`
  - 也就是master进程的网络地址
  - 默认是：127.0.0.1，只能用于单机。

- 通讯的port

- - `--master_port`
  - 也就是master进程的一个端口，要先确认这个端口没有被其他程序占用了哦。一般情况下用默认的就行
  - 默认是：29500

```bash
## Bash运行
# 假设我们在2台机器上运行，每台可用卡数是8
#    机器1：
python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 \
  --master_adderss $my_address --master_port $my_port main.py
#    机器2：
python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 8 \
  --master_adderss $my_address --master_port $my_port main.py
```

**小技巧**

```bash
# 假设我们只用4,5,6,7号卡
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 main.py
# 假如我们还有另外一个实验要跑，也就是同时跑两个不同实验。
#    这时，为避免master_port冲突，我们需要指定一个新的。这里我随便敲了一个。
CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 \
    --master_port 53453 main.py
```



### **mp.spawn**调用方式

PyTorch引入了torch.multiprocessing.spawn，可以使得单卡、DDP下的外部调用一致，即不用使用torch.distributed.launch。 python main.py一句话搞定DDP模式。

给一个mp.spawn的文档：[代码文档](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/_modules/torch/multiprocessing/spawn.html%23spawn)

下面给一个简单的demo：

```python
def demo_fn(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # lots of code.
    ...

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

**mp.spawn与launch各有利弊，请按照自己的情况选用。**
按照笔者个人经验，如果算法程序是提供给别人用的，那么mp.spawn更方便，因为不用解释launch的用法；但是如果是自己使用，launch更有利，因为你的内部程序会更简单，支持单卡、多卡DDP模式也更简单。



为什么只需要在master上加载就行 因为会进行复制

```python
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
```





1. 所谓的reduce，就是不同节点各有一份数据，把这些数据汇总到一起。在这里，我们规定各个节点上的这份数据有着相同的shape和data type，并规定汇总的方法是相加。**简而言之，就是把各个节点上的一份相同规范的数据相加到一起。**
2. 所谓的`all_reduce`，就是在reduce的基础上，把最终的结果发回到各个节点上。
3. 具体的all*reduce实现，要看具体的backend。流行的GPU backend NCCL，*all_reduce的实现就是使用了ring思想。













[[原创\][深度][PyTorch] DDP系列第一篇：入门教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/178402798)

[[原创\][深度][PyTorch] DDP系列第二篇：实现原理与源代码解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/187610959)