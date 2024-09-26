# Linux Ubuntu软件安装

## 概述

最近突然觉得需要记录一下软件的安装，以后可能能用的到，方便一点。



## 应用



### nodejs

这个是因为vscode中的leetcode插件需要nodejs环境

```bash
wget https://nodejs.org/dist/v16.1.0/node-v16.1.0-linux-x64.tar.xz
tar -xvf node-v16.1.0-linux-x64.tar.xz

#修改配置文件
vi .bashrc
export NODE_HOME="/home/USERNAME/node-v16.1.0-linux-x64/"
export PATH="$NODE_HOME/bin:$PATH"
#激活环境
source .bashrc
node -v
```



[薰风小记 | Ubuntu服务器安装Nodejs 16(无root权限) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/371610432)

leetcode插件要修改设置，主要是node的位置。

### anaconda

直接清华镜像源下载即可。

### nccl

[NVIDIA Collective Communications Library (NCCL) | NVIDIA Developer](https://developer.nvidia.com/nccl)

检测是否安装成功

```
python -c "import torch; print(torch.cuda.nccl.version())"
```



```sh
#add nccl path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xhao/software/nccl2/lib
export PATH=$PATH:/home/xhao/software/nccl2/include
```



### tensorflow v1



```sh
pip install tensorflow-gpu==1.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

#如果报错：

pip install protobuf==3.19.6 -i  https://pypi.tuna.tsinghua.edu.cn/simple

#cuda版本好像tensorflow只能适配到10.0
conda install cudatoolkit=10.0

```



### CUDA



[理清GPU、CUDA、CUDA Toolkit、cuDNN关系以及下载安装_cudatoolkit-CSDN博客](https://blog.csdn.net/qq_42406643/article/details/109545766)[无root 

[Linux安装CUDA 10.2 及 cudnn_linux cuda10 安装没有accept-CSDN博客](https://blog.csdn.net/hpqztsc/article/details/108516291)

https://blog.csdn.net/K_wenry/article/details/138350564

```
cp /data/xhao/tools/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/include/cudnn.h  /data/xhao/tools/cuda-11.8/include/
cp  /data/xhao/tools/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib/libcudnn*   /data/xhao/tools/cuda-11.8/lib64/

chmod a+r /data/xhao/tools/cuda-11.8/include/cudnn*.h
​
chmod a+r /data/xhao/tools/cuda-11.8/lib64/libcudnn*

```



### Git



下载路径：[Index of /pub/software/scm/git/ (kernel.org)](https://mirrors.edge.kernel.org/pub/software/scm/git/)

[Git 在Home目录下安装Git – CentOS 5 – 无ROOT权限|极客教程 (geek-docs.com)](https://geek-docs.com/git/git-questions/889_git_installing_git_in_home_directory_centos_5_no_root.html)



目前遇到了一个问题，但是报错了一个信息，git: 'remote-https' is not a git command. See 'git --help'.

有人说缺这个libcurl-devel

### Git-lfs



### RPM

先上这个网站搜索对应的版本：

```sh
rpm2cpio p7zip-16.02-20.el7.x86_64.rpm | cpio -idvm  #这个命令直接解压对应rpm文件到当前文件夹
```

注意要写入环境变量

bin文件使用$PATH

lib文件使用$LD_LIBRARY_PATH



安装opencv-python 报错 缺少libxet.so.6文件，使用下面网址进行安装。

[libXext-1.3.3-3.el7.x86_64.rpm – 查RPM (crpm.cn)](https://crpm.cn/libXext-1-3-3-3-el7-x86_64-rpm/)



### tmux安装

源码下载：

```bash
wget https://github.com/tmux/tmux/releases/download/2.2/tmux-2.2.tar.gz
wget https://github.com/libevent/libevent/releases/download/release-2.0.22-stable/libevent-2.0.22-stable.tar.gz
wget http://ftp.gnu.org/gnu/ncurses/ncurses-6.0.tar.gz
```

```bash
# libevent
./configure --prefix=/data/xhao/software/tmux --disable-shared
make && make install


# ncurses
./configure --prefix=/data/xhao/software/tmux
make && make install


./configure CFLAGS="-I/data/xhao/software/tmux/include -I/data/xhao/software/tmux/include/ncurses" LDFLAGS="-L/data/xhao/software/tmux/lib -L/data/xhao/software/tmux/include/ncurses -L/data/xhao/software/tmux/include" --prefix=/data/xhao/software/tmux/bin
make && make install
```

```bash
#环境变量设置
#将下面的语句添加到.bashrc中
export $PATH="/data/xhao/software/tmux/bin/bin:$PATH"
#重载环境
source .bashrc
```

[非root用户源码安装Tmux - 简书 (jianshu.com)](https://www.jianshu.com/p/f7f24b4b2625)
