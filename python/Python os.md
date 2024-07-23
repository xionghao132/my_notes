# Python os

## 概述

`os`模块提供的就是各种 Python 程序与操作系统进行交互的接口。

## os模块

* `os.getcwd()`：获取当前目录。

* `os.listdir()`:获取当前目录全部目录，包括文件。

* `os.mkdir(path)`:传入一个路径，创建一个目录。

* `os.chdir(path)`:切换工作目录。

```python
>>>os.chdir("d:/git")
>>>os.getcwd()
d:\\git
```

## os.path模块

* `os.path.isdir(path)`:判断传入的路径是否为目录。

* `os.path.isfile(path)`:判断传入的路径是否为文件。

* `os.path.join(str**)`:可以将多个传入路径组合为一个路径。

```python
>>>os.path.join("pthon","os","good")

python\\os\\good
```

* `os.path.abspath(path)`:将传入的路径规范化，返回一个绝对路径的字符串。

```python
>>>os.path.abspath("d:/python/os")

d:\\python\\os
```

* `os.path.basename(path)`:获取最后一个分隔符后面的内容。

```python
>>>os.path.basename("/python/os/good")

good
```

* `os.path.dirname(path)`:获取最后一个分隔符前面的内容。

* `os.path.split(path)`:以最后一个分解符为界，分成两个字符串，以元组的形式返回。

* `os.path.exists(path)`:判断路径所指向的位置是否存在。

[(1)Python os](https://zhuanlan.zhihu.com/p/150835193)