# Pathlib Path

## 概述

python中我们提取路径的方式一般都是使用os.path，很多时候都要进行切割等操作，Pathlib是python自带的库，可以使用更少的代码直接提取对应的路径。

## 使用

### 导包

```python
from pathlib import Path
path_str = Path(r"/data3/xh/demo.py")

#python文件中可以直接使用__file__
path_str=Path(__file__)
```

### 属性

```python
#提取文件名 demo.py
path_file_name = path_str.name

#提取父文件路径 /data3/xh

path_parent_path = path_str.parent

#提取文件后缀 .py
path_suffix = path_str.suffix

#提取无后缀的文件名 demo
path_only_name = path_str.stem

#更改文件后缀 /data3/xh/demo.json
path_suffix = path_str.with_suffix(".json")
```

### 方法

```python
#遍历目录
path_str = Path(r"/data3/xh/logs")
for path in path_str.iterdir():
    print(path)
#组合文件目录
path_str_join = path_str.joinpath("demo.py")
 
#是否是绝对路径
path_str.is_absolute()
pathlib 支持用 / 拼接路径。

#是否文件夹 or 文件
path_str.is_dir()
path_str.is_file()

#是否存在
path_str.exists()

#正则表达式查找目录下的文件
path_str = Path(r"/data3/xh/scripts")
print(path_str.glob('*.py'))    
    
```



[Python | Path 让文件路径提取变得简单(含代码)_python file path_HinGwenWoong的博客-CSDN博客](https://blog.csdn.net/hxj0323/article/details/113374539)