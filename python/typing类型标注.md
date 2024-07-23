# typing类型标注

## 概述

在python3.5之后，就引入了类型标注。尽管python在声明变量的时候不需要标注类型，但是对于我们开发过程中，传参和调用似乎不是那么方便，所以我们需要加入文档描述。类似于javascript一样，出现了加入类型的typescript。

## 例子

```python
def test(a:int, b:str) -> str:
    return str(a+b)
```

在函数的参数后面使用`:`，表明参数类型，在函数后面使用`->`，表明返回值的类型。

==注意：==如果我们在方法传入参数或者返回值与指定类型不对应的时候，不会报错，只会进行警告。原因是python本质上还是一门动态编译的语言。

## 常用数据类型

* int,long,float
* bool,str
* List,Tuple,Dict,Set
* Iterable,Iterator

```python
from typing import List,Turple,Dict
def test(a:List[int],b:Turple[int,int],Dict[str,str]) -> None:
```

[(21条消息) python3 限定方法参数 返回值 变量 类型_whatday的博客-CSDN博客_python3 指定函数返回值类型](https://blog.csdn.net/whatday/article/details/103475489#:~:text=python3 限定方法参数 返回值 变量 类型 1 int%2Clong%2Cfloat%3A 整型%2C长整形%2C浮点型,Tuple%2C Dict%2C Set%3A列表，元组，字典%2C 集合 4 Iterable%2CIterator%3A可迭代类型，迭代器类型 5 Generator：生成器类型)
