# \__init__.py

## 概述

在Python中，__init__.py文件用于将一个目录标记为Python的包。这个机制允许Python进行模块导入和组织代码的分层结构。尽管在Python 3.3及以上版本中，引入了隐式的命名空间包，这意味着在某些情况下即使没有__init__.py文件，目录也可以被视为包，但__init__.py文件仍然有其独特的作用和用途。

## 作用和用途

__init__.py的主要作用和用途包括：
将目录标记为Python包：这是__init__.py最基本的作用，它使得Python解释器知道该目录及其包含的文件应该被视为一个包。

初始化包：__init__.py可以包含包级别的初始化代码。这意味着当包被导入时，__init__.py中的代码将被执行。这对于设置包所需的全局状态或者执行仅需在包首次导入时运行的代码很有用。

控制包的导入：通过在__init__.py文件中导入子模块或包内其他模块，你可以定制from package import *时哪些模块被导入的行为。虽然这种做法不推荐（因为显式导入通常更清晰），但它可以用来简化客户端代码的导入语句。

包的命名空间管理：__init__.py文件可以用来组织包的命名空间。通过在这个文件中导入函数、类或其他模块，你可以提供一个经过精心设计的对外接口，使得包的结构对用户更加透明。



## 模块（[module](https://so.csdn.net/so/search?q=module&spm=1001.2101.3001.7020)）

Python中的任何 `.py` 文件都可以称为一个**模块**（module），模块可以用来组织函数、类等代码，然后在其他的程序中引入这些模块中的功能，有利于代码的管理与提高重用性。使用`import`语句即可引入模块。

- `import module_name`

引入模块，同时引入多个模块使用逗号隔开，然后使用如下的方法调用模块中的方法、类等：

```python
module_name.function
module_name.class
```

## 包（package）

包（package）在python中是用于管理模块文件夹，一个文件夹能够成为一个包，需要包含一个名为 __init__.py 的文件，这个文件可以是空的，它作为这个文件夹是一个包的标识。

比如在main.py所在目录下创建一个文件夹名为 myPackage，里边需要有一个 __init_.py 文件，然后我们再在 myPackage 中创建一个 myModule2.py：

```
|--- Project
   |--- main.py
   |--- myPackage
      |--- __init__.py
      |--- myModule2.py
```



## 实例

```text
my_package/
│
├── __init__.py
├── submodule1.py
└── submodule2.py
```

在`my_package/__init__.py`中，我们可以初始化包或者导入特定的模块以简化用户的导入语句：

```python
# my_package/__init__.py
from .submodule1 import ClassA
from .submodule2 import functionB


```

本质上，运行的代码在哪层目录，交互的路径就在对应的层。

注意：**使用import语法，但必须记住最后一个属性必须是子包或模块，它不应该是任何函数或类名**。



不能直接导入文件夹，需要在对应文件夹下的\__init__.py里面导入对应的包

https://blog.csdn.net/yangweipeng708/article/details/136857164



## Python -m



在Python中，使用-m参数可以执行一个模块作为脚本。它是用于从命令行直接运行一个Python模块的标志。这种方式具有以下几个方面的作用：

* 直接执行模块代码: 使用python -m命令可以直接在命令行中执行一个Python模块，而不需要编写额外的启动脚本。这对于简单的脚本或工具非常方便，因为它们可以作为独立的可执行文件运行。

* 模块自测试: 当一个模块被设计为既可以作为库使用，又可以作为独立脚本运行时，可以将自测试代码放在__main__函数中，并使用python -m来运行该模块以进行测试。这样可以确保模块在被导入时正常运行，同时也能够通过直接执行来验证其功能。
  

直接使用Python -m 如何在里面加个main函数进行测试，不需要额外创建main.py文件