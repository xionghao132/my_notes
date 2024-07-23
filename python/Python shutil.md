# Python shutil

## 概述

`os`模块提供了对文件目录常用的操作，`shutil`模块可以对文件目录进行复制、移动、删除、压缩、解压等操作。

## 常用方法

* `shutil.copy(src,dst)` ：复制文件

* ` shutil.copytree(src,dst)`：复制文件夹 ，只能复制空文件夹

*  `shutil.move(src,dst)`：移动文件或文件夹

* `shutil.rmtree(src)`：删除文件夹，可以递归删除非空文件

## 压缩与解压

`shutil `模块对压缩包的处理是调用` ZipFile` 和 `TarFile`这两个模块来进行的，因此需要导入这两个模块

* `zipobj.write()`：创建压缩包

```python
import zipfile
import os
file_list = os.listdir(os.getcwd())
# 'w'写入
with zipfile.ZipFile(r"my.zip", "w") as zipobj:
    for file in file_list:
        zipobj.write(file)

```

* `zipobj.namelist()`：读取压缩包文件信息

```python
with zipfile.ZipFile('my.zip','r') as zipobj:
    print(zipobj.namelist())
```

* `zipobj.extract()`：将压缩包中的单个文件，解压出来

```python
dst = r"D:\file" #目标目录
with zipfile.ZipFile('my.zip','r') as zipobj:
    zipobj.extract('data',dst)
```
==注意：==目标目录不存在可以自动创建

* `zipobj.extractall()`：将压缩包中所有文件都解压出来

```python
dst = r"D:\file" #目标目录
with zipfile.ZipFile("我创建的压缩包.zip", "r") as zipobj:
    zipobj.extractall(dst)
```

[(1) Python shutil](https://blog.csdn.net/weixin_41261833/article/details/108050152)

