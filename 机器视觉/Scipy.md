# Scipy

## 概述

`SciPy` 是一个开源的 Python 算法库和数学工具包。

`Scipy` 是基于 Numpy 的科学计算库，用于数学、科学、工程学等领域，很多有一些高阶抽象和物理模型需要使用` Scipy`。

`SciPy` 包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。



## 模块列表

| 模块名            | 功能               | 参考文档                                                     |
| :---------------- | :----------------- | :----------------------------------------------------------- |
| scipy.cluster     | 向量量化           | [cluster API](https://docs.scipy.org/doc/scipy/reference/cluster.html) |
| scipy.constants   | 数学常量           | [constants API](https://docs.scipy.org/doc/scipy/reference/constants.html) |
| scipy.fft         | 快速傅里叶变换     | [fft API](https://docs.scipy.org/doc/scipy/reference/fft.html) |
| scipy.integrate   | 积分               | [integrate API](https://docs.scipy.org/doc/scipy/reference/integrate.html) |
| scipy.interpolate | 插值               | [interpolate API](https://docs.scipy.org/doc/scipy/reference/interpolate.html) |
| scipy.io          | 数据输入输出       | [io API](https://docs.scipy.org/doc/scipy/reference/io.html) |
| scipy.linalg      | 线性代数           | [linalg API](https://docs.scipy.org/doc/scipy/reference/linalg.html) |
| scipy.misc        | 图像处理           | [misc API](https://docs.scipy.org/doc/scipy/reference/misc.html) |
| scipy.ndimage     | N 维图像           | [ndimage API](https://docs.scipy.org/doc/scipy/reference/ndimage.html) |
| scipy.odr         | 正交距离回归       | [odr API](https://docs.scipy.org/doc/scipy/reference/odr.html) |
| scipy.optimize    | 优化算法           | [optimize API](https://docs.scipy.org/doc/scipy/reference/optimize.html) |
| scipy.signal      | 信号处理           | [signal API](https://docs.scipy.org/doc/scipy/reference/signal.html) |
| scipy.sparse      | 稀疏矩阵           | [sparse API](https://docs.scipy.org/doc/scipy/reference/sparse.html) |
| scipy.spatial     | 空间数据结构和算法 | [spatial API](https://docs.scipy.org/doc/scipy/reference/spatial.html) |
| scipy.special     | 特殊数学函数       | [special API](https://docs.scipy.org/doc/scipy/reference/special.html) |
| scipy.stats       | 统计函数           | [stats.mstats API](https://docs.scipy.org/doc/scipy/reference/stats.mstats.html) |



## 常量模块

```python
from scipy import constants

print(constants.yotta)    #1e+24
print(constants.zetta)    #1e+21
print(constants.exa)      #1e+18
print(constants.peta)     #1000000000000000.0
print(constants.tera)     #1000000000000.0
print(constants.giga)     #1000000000.0
print(constants.mega)     #1000000.0
print(constants.kilo)     #1000.0
print(constants.hecto)    #100.0
print(constants.deka)     #10.0
print(constants.deci)     #0.1
print(constants.centi)    #0.01
print(constants.milli)    #0.001
print(constants.micro)    #1e-06
print(constants.nano)     #1e-09
print(constants.pico)     #1e-12
print(constants.femto)    #1e-15
print(constants.atto)     #1e-18
print(constants.zepto)    #1e-21
```



## 优化器

```python
from scipy.optimize import root
from math import cos

def eqn(x):
  return x + cos(x)

myroot = root(eqn, 0)

print(myroot)
```



## 稀疏矩阵

> 这个比较重要，面对大数据存储的可以使用稀疏矩阵去存储，节省空间。

- `CSC `- 压缩稀疏列**（Compressed Sparse Column）**，按列压缩。
- `CSR` - 压缩稀疏行**（Compressed Sparse Row）**，按行压缩。

```python
import numpy as np
from scipy.sparse import csr_matrix

arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])

print(csr_matrix(arr))

print(csr_matrix(arr).count_nonzero()) #可以统计非零数据

csr_matrix(arr).sum_duplicates() #删除重复项
newarr = csr_matrix(arr).tocsc() #转化为csc
```

[【Scipy学习】Scipy中稀疏矩阵用法解析（sp.csr_matrix；sp.csc_matrix；sp.coo_matrix）_一穷二白到年薪百万的博客-CSDN博客](https://blog.csdn.net/zfhsfdhdfajhsr/article/details/116934577?ydreferer=aHR0cHM6Ly93d3cuYmluZy5jb20v)





[Scipy 显著性检验 | 菜鸟教程 (runoob.com)](https://www.runoob.com/scipy/scipy-significance-tests.html)