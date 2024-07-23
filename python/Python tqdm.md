# Python tqdm

## 概述

主要用于定义进度条，有比较好的可视化效果。尤其是在我们训练网络的时候可以使用，但是它的缺点是会降低一些训练速度。

## 模式

`tqdm`主要分为两种模式：

1. 基于迭代对象运行

```python
from tqdm import tqdm  #导入依赖
import time

list=[1,2,3]
for i in tqdm(list):
    time.sleep(0.5)   #仅为了展示而休眠

for imaget,target in tqdm(dataloader): #深度学习训练过程中可以用
    ...
    
for i in tqdm(range(100), desc='Processing'): #desc类似于前面的标签
    time.sleep(0.05)

dic = ['a', 'b', 'c', 'd', 'e']
pbar = tqdm(dic)
for i in pbar:
    pbar.set_description('Processing '+i)
    time.sleep(0.2)
```

2. 手动更新

```python
import time
from tqdm import tqdm

with tqdm(total=200) as pbar:
    pbar.set_description('Processing:')
    # total表示总的项目, 循环的次数20*10(每次更新数目) = 200(total)
    for i in range(20):
        # 进行动作, 这里是过0.1s
        time.sleep(0.1)
        # 进行进度更新, 这里设置10个
        pbar.update(10)
```

==注意：==导入依赖直接写`import tqdm`可能会报错`module object is not callable`，修改成上面的示例即可。

## 可迭代对象

当出现上面的错误时候，我们可以检查一下，我们放入的对象是否为可迭代对象。

```python
from collections import Iterable  
isinstance('abcde',Iterable)  #判断对象是否为可迭代对象
```

[(1)python tqdm](https://zhuanlan.zhihu.com/p/163613814)

