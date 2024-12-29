# Conda和pip使用指南

## Conda环境管理

### 创建环境

```sh
# 创建一个名为python34的环境，指定Python版本是3.5（不用管是3.5.x，conda会为我们自动寻找3.５.x中的最新版本）
conda create --name pytorch python=3.5
```

### 激活环境

```sh
conda activate pytorch  #windows
```

### 返回主环境

```sh
# 返回
deactivate pytorch # for Windows
```

### 删除环境

```sh
# 删除一个已有的环境
conda remove --name pytorch --all
```

### 复制环境

```sh
#从一个已有的环境old_env中复制
conda create -n new_env --clone old_env
```



### 查看系统中的所有环境

> 用户安装的不同Python环境会放在`~/anaconda/envs`目录下。查看当前系统中已经安装了哪些环境，使用`conda info -e`。

## Conda的包管理

### 安装库

```sh
conda install numpy
```

### 查看已经安装的库

```sh
# 查看已经安装的packages
conda list
```

### 查看某个环境的已安装包

```sh
# 查看某个指定环境的已安装包
conda list -n pytorch
```

### 搜索package的信息

```sh
# 查找package信息
conda search numpy
```

### 安装package到指定的环境

```sh
# 安装package
conda install -n pytorch numpy
```

### 更新package

```sh
# 更新package
conda update -n pytorch numpy
```

### 删除package

```sh
# 删除package
conda remove -n pytorch numpy
```

### 更新conda

```sh
# 更新conda，保持conda最新
conda update conda
```

### 更新anaconda

```sh
# 更新anaconda
conda update anaconda
```

### 更新python

```sh
conda update python
```

### 设置国内镜像

```sh
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 
conda config --set show_channel_urls yes
```

### 配置文件

`~/.condarc`

```sh
envs_dirs:
  - /data/conda/envs
pkgs_dirs:
  - /data/conda/pkgs
```

直接修改保存就行，下次创建的内容会直接在新设置的位置



### 导出环境文件

环境迁移

```sh
#首先要激活对应的环境
conda env export > environment.yml

conda env create --file environment.yml
```



## pip使用指南

###  卸载已安装的库

```sh
pip uninstall pillow
```

### 列出已经安装的库

```sh
pip list
```

### pip升级

```sh
pip install --upgrade pip
```

### pip安转

```sh
pip install django==1.8
```

