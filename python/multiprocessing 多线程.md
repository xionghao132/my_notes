# multiprocessing 多线程

## 概述

`multiprocessing`模块支持使用类似于`threading`模块的API生成进程。`multiprocessing`模块提供了本地和远程计算机的并行处理能力，并且通过使用创建子进程，有效地避开了全局解释器锁**（GIL）**。因此，`multiprocessing`模块允许程序员充分利用机器上的多个处理器。目前，它可以在`Unix`和`Windows`上运行。



多进程只有`cpu`核数个进程才是真正并行的。超过`cpu`核数个进程都是`cpu`分配时间片实现进程之间的频繁切换，看起来是并行，实际上是并发，也就是同一时间段内的一起运行。而进程的切换相对线程切换消耗资源过大，所以不适应用`multiprocessing`开很多个进程。

由于**GIL**的存在，`python`中的多线程其实并不是真正的多线程，如果想要充分地使用多核`CPU`的资源，在`python`中大部分情况需要使用多进程。

**进程Process的创建远远大于线程Thread创建占用的资源**

## 进程

`multiprocessing.Process`是创建一个进程。

`join`目的是让主线程暂停运行，等对应的程序运行完成。

- `target`：在进程中被执行的函数
- `args`：向被执行函数传递的参数

```python
# importing the multiprocessing module 
import multiprocessing 

def print_cube(num): 
    print("Cube: {}".format(num * num * num)) 

def print_square(num): 
    print("Square: {}".format(num * num)) 

if __name__ == "__main__": 
    # creating processes 
    p1 = multiprocessing.Process(target=print_square, args=(10, )) 
    p2 = multiprocessing.Process(target=print_cube, args=(10, )) 

    # starting process 1&2
    p1.start() 
    p2.start() 

    # wait until process 1&2 is finished 
    p1.join() 
    p2.join() 

    # both processes finished 
    print("Done!")
```

## 进程池



* **map(func,iterable)** — `Pool`类中的`map`方法，与内置的map函数用法基本一致，它会使进程阻塞直到结果返回
* **imap(func,iterable)**会返回一个迭代器，迭代器里面是按顺序返回的函数执行结果，可配合tqdm使用，显示进度。

注意：虽然第二个参数是一个迭代器，但在实际使用中，必须在整个队列都就绪后，程序才会运行子进程。

* **close()** — 关闭进程池**（pool）**，使其不在接受新的任务。
* **join()** — 主进程阻塞等待子进程的退出，` join`方法要在`close`或`terminate`之后使用。

```python
# 导入进程模块
import multiprocessing
 
# 最多允许3个进程同时运行
#这里设置允许同时运行的的进程数量要考虑机器cpu的数量，进程的数量最好别小于cpu的数量，
pool = multiprocessing.Pool(processes = 3)

# 进程池作为上下文管理器
with Pool(num_workers) as p:
    ......
```







需要注意的是，在Windows上要想使用进程模块，就必须把有关进程的代码写在**if _\_name\_\_ == ‘\_\_main\_\_’** 内，否则在Windows下使用进程模块会产生异常。`Unix/Linux`下则不需要。



配合偏函数`partial`使用。

`partial()`主要是固定一部分参数，然后

```python
rom multiprocessing import Pool
from functools import partial

# 需要重复执行的函数
def func(a, b, c):
    pass

pool = Pool(4)

# 作为每次执行的输入的参数迭代器
cs = [...] # iterable[, chunksize]
partial_func=partial(func, a = 1, b = 2)
pool.map(partial_func, c = cs)
pool.close()
pool.join() 
```

```python
from multiprocessing import Pool

from tqdm import tqdm

# 需要重复执行的函数
def func(*args, **kargs):
    return None

pool = Pool(4)

# 作为每次执行的输入的参数迭代器
parameters = [...] # iterable[, chunksize]

results = pool.imap(func, parameters)

for result in tqdm(results):
    print(result)

pool.close()
pool.join() 
```