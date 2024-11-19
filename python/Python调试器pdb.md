# Python调试器pdb

## 方法

- **非侵入式方法**（不用额外修改源代码，在命令行下直接运行就能调试）

```bash
python3 -m pdb filename.py
```

- **侵入式方法**（需要在被调试的代码中添加一行代码然后再正常运行代码）

```python3
import pdb;pdb.set_trace()

breakpoint()
```

如果代码在运行的时候，忘记删除breakpint()

``` python
import builtins; builtins.breakpoint = lambda: None  # 临时禁用 breakpoint()
```



## 命令

* 查看源码

```sh
l    #查看当前位置前后11行源代码（多次会翻页）
ll   #查看当前函数或框架的所有源代码
w    #找到运行到了哪一行
```

* 添加断点

```sh
b  
b lineno          #看着文件行就行
b filename:lineno    
b functionname
tbread            #是添加临时断点 参数同b
```

参数：

> filename文件名，断点添加到哪个文件，如test.py
> lineno断点添加到哪一行
> function：函数名，在该函数执行的第一行设置断点

说明：

> 1.不带参数表示查看断点设置
> 2.带参则在指定位置设置一个断点

* 清除断点

```sh
cl
cl filename:lineno
cl bpnumber [bpnumber ...]
```

参数：

> bpnumber 断点序号（多个以空格分隔）

说明：

> 1.不带参数用于清除所有断点，会提示确认（包括临时断点）
> 2.带参数则清除指定文件行或当前文件指定序号的断点

* 打印变量值

```sh
p expression
```

参数：

> expression Python表达式

* 调试命令

```sh
s    #执行下一行，能够进入函数体
n    #执行下一行，不会进入函数体
r    #执行下一行，在函数中时直接直行道函数返回处

c    #持续执行到下一个断点
unt lineno   #持续执行到指定行
j lineno    #直接跳转到指定行，被跳过的代码不执行
```

* 打印变量类型

```sh
whatis expression
```

说明：

> 打印表达式的类型，常用来打印变量值

* 启动交互解释器

```sh
interact
```

说明：

> 启动一个python的交互式解释器，使用当前代码的全局命名空间（使用ctrl+d返回pdb）

* 打印堆栈

```sh
w
```

说明：

> 打印堆栈信息，最新的帧在最底部。箭头表示当前帧。

* 退出pdb

```
q
```





[10分钟教程掌握Python调试器pdb - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/37294138)