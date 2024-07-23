# Python glob

## 概述

`glob`模块用来查找**文件目录和文件**，并将搜索的到的结果返回到一个列表中。

我们在使用`pytorch`导入自己的`DataSet`的时候，使用这个模块很方便，因为数据的文件名很容易匹配。

## 通配符

* $*$匹配任意字符
* $？$匹配一个字符
* $[]$匹配指定范围内的字符，$[0-9]$匹配数字，$[a-zA-Z]$匹配英文字母。

## 实例

> `glob.glob()` 函数

获取目录下匹配成功的所有文件和目录，并返回一个列表

```python
import glob
def find(data_path):
	imgs_path=glob.glob(os.path.join(data_path, 'image/*.png')) # 比如数据格式都是image/*.png
    for i,p in enumerate(imgs_path):
        image=imgs_path[i]
        label=image.replace('image','label') #一般标签和训练数据在一层目录
```

使用`os.listdir(path)`获取path目录下所有文件和目录。

