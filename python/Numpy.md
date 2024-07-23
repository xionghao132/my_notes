# Numpy

## 概述

**NumPy** 是一个运行速度非常快的数学库，主要用于数组计算

**NumPy** 通常与 **SciPy**和 **Matplotlib**一起使用， 这种组合广泛用于替代 MatLab，是一个强大的科学计算环境，有助于我们通过 Python 学习数据科学或者机器学习。

## Ndarray对象

```python
import numpy as np 
a = np.array([1,2,3])  
print (a)
```

## 常见的数据类型

一般是传入的参数`dtype`

```python
np.int32
np.float
np.uint32 #无符号32位整数

a = np.array([(10,),(20,),(30,)], dtype = np.int32) 
```

## 数组属性

```python
ndarray.shape                   #数组的维度，对于矩阵，n 行 m 列
ndarray.size                    #数组大小
ndarray.nmin                    #返回数组的秩
ndarray.T                       #矩阵转置
```

## 创建数组

* 重新创建数组

```python
x = np.empty([3,2], dtype = int) 
y = np.zeros((5,), dtype = int) 
x = np.ones([2,2], dtype = int)
```

* 从已有数组中创建数组

```python
x =  [1,2,3] 
a = np.asarray(x)  
```

* 数值范围创建数组

```python
a = np.linspace(1,1,10) #start end num
a                       #[1]*10
```

切片索引，操作和`list`类似。

## 广播机制

* `ndarray`的乘法是类似数学中的空心乘法，矩阵对应位置相乘，如果是点乘，使用`np.dot`

```python
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b #[ 10  40  90 160]
```

* 数组形状不同，自动触发广播机制

```python
a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([0,1,2]) #自动向下扩展	并且行向量相同
a+b                
```

> [[ 0  1  2]
>  [10 11 12]
>  [20 21 22]
>  [30 31 32]]

## 数组操作

```python
b = a.reshape(4,2)           #改变数组形状
a.flatten()                  #展平数组
np.transpose(a)              #翻转数组 
```

## 数学函数

* 三角函数

```python
np.sin(a*np.pi/180)
np.cos(a*np.pi/180)
np.tan(a*np.pi/180)
```

> arcsin，arccos，和 arctan 函数返回给定角度的 sin，cos 和 tan 的反三角函数。

* 舍入函数

```python
np.round()
np.floor()
np.ceil()
```

* 开方

```
np.sqrt()
```

## 算术函数

```python
np.add()
np.subtract()
np.matmul()
np.divide()
np.mod()
np.reciprocal()   #返回倒数]
np.power()
```

## 统计函数

* 最大最小值

```python
a = np.array([[3,7,5],[8,4,3],[2,4,9]])  
print (np.amin(a,1))   #[3 3 2]  #行
np.amax(a, axis =  0)  #[8 7 9]  #列 
np.amax(a)             #9 不加入轴就是全数组搜索（下面同理）
```

==注意：==`axis`表示该轴变化，其他轴不变

* 极值

```
numpy.ptp()         #函数计算数组中元素最大值与最小值的差
```

* 中位数，平均数，均值，标准差，方差

```python
np.median()
np.average()   #也可以加入权重数组，就是加权平均数
np.mean()
np.std() 	   #std = sqrt(mean((x - x.mean())**2))
np.var()
```

## NumPy 排序、条件刷选函数

```python
np.sort(a, axis =  0) #按列进行排序，默认按行进行排序
np.argsort()          #函数返回的是数组值从小到大的索引值。
np.argmax()           #函数分别沿给定轴返回最大元素的索引。
np.argmin()           #函数分别沿给定轴返回最小元素的索引。
y = np.where(x >  3)  #返回输入数组中满足给定条件的元素的索引
```

## 线性代数

**NumPy** 提供了线性代数函数库 **linalg**，该库包含了线性代数所需的所有功能

```python
np.dot()                   #两个数组的乘积matmul，需要满足前一个数组的列等于后一个数组的行
np.vdot()				   #函数是两个向量的点积
np.trace()                 #求矩阵的迹
np.linalg.det()            #计算输入矩阵的行列式
np.linalg.solve()          #给出了矩阵形式的线性方程的解
numpy.linalg.inv()         #求矩阵的逆
numpy.linalg.eig()         #返回两个值，第一个是特征值矩阵，第二个是特征向量矩阵
```

==注意：==`numpy`中数组和矩阵的使用区别不是很大，尽量只调用函数而不是使用符号。
