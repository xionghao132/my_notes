# Colab的使用

[![Open In Colab](D:\my_ty_file\images\colab-badge.svg)](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

google colab 提供免费的GPU资源，对于学生党来说太有用处了。

==注意:==使用前提需要有科学上网的账号，其次要注册有谷歌的帐号。

## 设置GPU

点击代码执行程序->更改运行类型  选择硬件加速类型为GPU，通过下面命令可以查看GPU类型。

```sh
!nvidia-smi
```

## Colab 链接Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

弹出连接完成提示的操作后就可以完成连接,查看文件目录可以看到`drive`目录了。这个时候就可以查看goole Drive的数据了。

==作用：==数据较大的时候，可以将数据上传到`Drive`中，然后在colab中可以使用。

## 代码块

`colab`代码框本质是`linux`的输入框。

正常输入框中是`python`代码，前面输入`!`就输入`linux`命令。

## 安装库

* 查看库

```sh
!pip list
```

* 安装包

```sh
!pip install 
```

## 目录

* 查看当前目录

```sh
！pwd
```

* 切换目录

```sh
%cd ..
```



