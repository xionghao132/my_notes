# python setup.py

## 概述

平常我们习惯了使用 pip 来安装一些第三方模块，这个安装过程之所以简单，是因为模块开发者为我们默默地为我们做了所有繁杂的工作，而这个过程就是 `打包`。

打包，就是将你的源代码进一步封装，并且将所有的项目部署工作都事先安排好，这样使用者拿到后即装即用，不用再操心如何部署的问题。

不管你是在工作中，还是业余准备自己写一个可以上传到` PyPI` 的项目，你都要学会如何打包你的项目。

Python 发展了这么些年了，项目打包工具也已经很成熟了有 `distutils` 、`distutils2`、`setuptools`等等

## distutils

它是 `Python` 官方开发的一个分发打包工具，所有后续的打包工具，全部都是基于它进行开发的。

`distutils` 的精髓在于编写 setup.py，它是模块分发与安装的指导文件。

我们经常用它来进行模块的安装。

```sh
python setup.py install
```

## setuptools

`setuptools` 是 `distutils `增强版，不包括在标准库中。其扩展了很多功能，能够帮助开发者更好的创建和分发` Python` 包。大部分` Python` 用户都会使用更先进的` setuptools` 模块。

[python打包分发工具：setuptools - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/460233022)

## setup.py使用

正常情况下，我们都是通过以上构建的源码包或者二进制包进行模块的安装。

但在编写` setup.py` 的过程中，可能不能一步到位，需要多次调试，这时候如何测试自己写的 `setup.py `文件是可用的呢？

这时候你可以使用这条命令，它会将你的模块安装至系统全局环境中

```sh
python setup.py install
```

如若你的项目还处于开发阶段，频繁的安装模块，也是一个麻烦事。

这时候你可以使用这条命令安装，该方法不会真正的安装包，而是在系统环境中创建一个软链接指向包实际所在目录。这边在修改包之后不用再安装就能生效，便于调试。

```sh
python setup.py develop
```

这个是`DrQA-main`的`setup.py`

```
#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys

with open('README.md', encoding='utf8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf8') as f:
    reqs = f.read()

setup(
    name='drqa',
    version='0.1.0',
    description='Reading Wikipedia to Answer Open-Domain Questions',
    long_description=readme,
    license=license,
    python_requires='>=3.5', #对python版本的限制
    packages=find_packages(exclude=('data')), #找到当前目录下的一些包进行安装 还能排除目录文件
    install_requires=reqs.strip().split('\n'), #这个就是要安装的包 list
)

```



## 发布包到PyPi

`PyPi `（Python Package Index）上，它是 Python 官方维护的第三方包仓库，用于统一存储和管理开发者发布的 Python 包。

如果要发布自己的包，需要先到 pypi 上注册账号。然后创建 `~/.pypirc` 文件，此文件中配置 PyPI 访问地址和账号。如的.pypirc文件内容请根据自己的账号来修改。

典型的 `.pypirc` 文件

```text
[distutils]
index-servers = pypi

[pypi]
username:xxx
password:xxx
```



然后使用这条命令进行信息注册，完成后，你可以在 PyPi 上看到项目信息。

```sh
python setup.py register
```

注册完了后，你还要上传源码包，别人才使用下载安装

```sh
python setup.py upload
```

或者也可以使用 `twine` 工具注册上传，它是一个专门用于与 `pypi` 进行交互的工具，详情可以参考官网：[https://www.ctolib.com/twine.html，这里不详细讲了。](https://link.zhihu.com/?target=https%3A//www.ctolib.com/twine.html%EF%BC%8C%E8%BF%99%E9%87%8C%E4%B8%8D%E8%AF%A6%E7%BB%86%E8%AE%B2%E4%BA%86%E3%80%82)







[花了两天，终于把 Python 的 setup.py 给整明白了 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/276461821)