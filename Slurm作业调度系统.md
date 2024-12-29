# Slurm作业调度系统

## 概述

Slurm是一个广泛使用的开源作业调度系统，专为Linux集群设计，无论是大型还是小型。

## 系统常用操作

- 交互模式:可I/O或信号交互,`srun`命令
- 批处理模式:编写提交作业脚本,`sbathc`命令
- 分配模式:预分配资源,可交互`salloc`命令

### 节点操作

查看节点与分区

```sh
sinfo
```



查看队列

```sh
squeue -a
```

| 关键词    | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| PARTITION | 分区名，大型集群为了方便管理，会将节点划分为不同的分区设置不同权限 |
| AVAIL     | 可用状态：up 可用；down 不可用                               |
| TIMELIMIT | 该分区的作业最大运行时长限制, 30:00 表示30分钟，如果是2-00:00:00表示2天，如果是infinite表示不限时间 |
| NODES     | 数量                                                         |
| STATE     | 状态：drain: 排空状态，表示该类结点不再分配到其他；idle: 空闲状态；alloc: 被分配状态;mix:部分被占用，但是仍有可用资源 |

查看作业

```sh
scontrol show job jobid
squeue -u username
```

申请资源

```sh
# 新建交互式作业,先占用资源,成功后返回任务JOBID
salloc -p gpu --gres=gpu:1 bash

# 通过队列可以看到任务分配的节点,进入节点
$ ssh [NODE_ID]
```



### 提交作业



提交单个作业

```sh
srun hostname
srun -N 2 -l hostname
```



提交脚本

`Slurm`允许您使用 `sbatch` 命令提交脚本作业。例如，要提交一个名为 `job_script.sh` 的作业脚本，您可以使用命令 `sbatch job_script.sh`。

> vim run.slurm

```python
#!/bin/bash
#SBATCH -J test_2023-03-22
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -t 3:00
sleep 100
echo test_2021-03-22
```



```sh
chmod 775 run.slurm
sbatch run.slurm
```



使用`conda`环境

```sh
#!/bin/bash
#SBATCH --job-name=myjob            # 作业名称
#SBATCH --output=myjob.out          # 标准输出和错误日志
#SBATCH --error=myjob.err           # 错误日志文件
#SBATCH --ntasks=1                  # 运行的任务数
#SBATCH --time=01:00:00             # 运行时间
#SBATCH --partition=compute         # 作业提交的分区

# 加载Conda
source ~/miniconda3/etc/profile.d/conda.sh

# 激活环境
conda activate myenv

# 运行命令
python my_script.py
```



更新任务

```
scontrol update jobid=938 partition=gpu gres=gpu:1
```



取消作业

```
# 取消具体某作业id对应作业
scancel ${jobid}

# 取消某用户的作业
scancel -u ${username}
```



## 学校系统操作

```sh
srun -p batch -N 1 -n 6 -t 4320 --gres=gpu:TeslaV100-SXM2-32GB:1 --mem=60G --pty /bin/bash
srun -p batch -N 1 -n 6 -t 4320 --gres=gpu:NVIDIAA100-PCIE-40GB:1 --mem=60G --pty /bin/bash
```



```txt
-p <使用的分区>
-N <节点数量>
-n 指定要运行的任务数，每个任务默认一个核心
--gres=gpu:<单节点 GPU 卡数>
-t <最长运行时间>
--mem 使用的内存
--qos=<使用的 QoS>
```



`sbatch`操作

```
#!/bin/bash

#SBATCH -J xhao
#SBATCH -n 6
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
#SBATCH -o log_%j.log # 请替换你想要输出的日志名
#SBATCH -e log_%j.err # 请替换你想要输出的日志名
#SBATCH --mem=60G
#SBATCH --time=72:00:00


source /public/home/wlchen/miniconda3/bin/activate
# conda init
conda activate xh # 请替换为正确的conda环境

# 下面你正常的运行

sh test.sh

# 注意！！！你还可以通过下面这种方式一次性跑多个sh

# list="file_path1 file_path2 file_path3"
# # 使用for循环遍历列表中的每个file_path
# for file_path in $list; do
#   echo "开始运行: $file_path"
#   sh $file_path
# done
```

[Slurm入门指南：Linux集群作业调度与配置](https://blog.csdn.net/lejun_wang1984/article/details/135180652)