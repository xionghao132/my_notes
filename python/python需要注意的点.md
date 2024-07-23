# Python需要注意的点

## 基础语法篇

### 起步

python2与python3的区别？

在Python 2中，无需将要打印的内容放在括号内。从技术上说，Python 3中的print 是一个函数，因此括号必不可少。

在Python 2中，整数除法的结果只包含整数部分，小数部分被删除。请注意，计算整数结果时，采取的方式不是四舍五入，而是将小数部分直 接删除。 在Python 2中，若要避免这种情况，务必确保至少有一个操作数为浮点数，这样结果也将为浮点数：

python2创建类需要加入object

```python
>>> 3 / 2
1
>>> 3.0 / 2
1.5
>>> 3 / 2.0
1.5
>>> 3.0 / 2.0
1.5
```

python3中/表示浮点数除法，//表示整除

#### linux下安装python3

1. 使用wget下载包
2. docker拉取conda镜像

#### windows下安装python3

直接安装conda比较方便

### 变量和简单数据类型

命名规则时小写字母加下划线的方式，不是使用的驼峰规则

变量前面不需要加声明

不可变类型

[Python中的不可变对象类型与可变对象类型 - listenviolet - 博客园 (cnblogs.com)](https://www.cnblogs.com/shiyublog/p/10809953.html#:~:text=Python中的不可变对象类型与可变对象类型 1 1. 对象类型 不可变 (immutable)对象类型 int float,7 7. 总结 ... 8 8. 参数传递 )



数值(整型，浮点型)，布尔型，字符串，元组属于值类型，本身不允许被修改(不可变类型)，数值的修改实际上是让变量指向了一个新的对象(新创建的对象)，所以不会发生共享内存问题。。原始对象被Python的GC回收。

str.strip():消除字符串两端空格

**表示乘方

结果包含的小数位数可能是不确定的：

```python
>>> 0.2 + 0.1
0.30000000000000004
```

整形与字符串不能直接相加串联起来

需要将整形数字使用str()转化为字符串然后才能相加

###  列表简介

list.pop()可以删除任意位置的索引元素

list.sort()永久性排序

sorted()临时性排序

reverse=True

list.reverse()

###  操作列表

列表解析 将for 循环和创建新元素的代码合并成一行，并自动 附加新元素

```python
squares = [value**2 for value in range(1,11)]
print(squares)
```

列表复制

```python
list1=list    #浅拷贝
list1=list[:] #深拷贝
```

修改元组的操作是被禁止的

虽然不能修改元组的元素，但可以给存储元组的变量赋值

### if 语句

要判断特定的值是否已包含在列表中，可使用关键字in

```
>> requested_toppings = ['mushrooms', 'onions', 'pineapple'] ❶ >>> 'mushrooms' in requested_toppings
True 
>>> 'pepperoni' in requested_toppings
False
```

and or

### 字典

```
for k, v in user_0.items()
```

```
del dict['sf']
dict.pop('sf')
```

 for name in favorite_languages.keys(): print(name.title())

```
for name in sorted(favorite_languages.keys()):
print(name.title() + ", thank you for taking the poll.")
```

```
for language in set(favorite_languages.values()):
print(language.title())
```

### 用户输入和while 循环

###  函数

 传递任意数量的实参

```
def make_pizza(*toppings):
"""打印顾客点的所有配料"""
print(toppings)
make_pizza('pepperoni')
make_pizza('mushrooms', 'green peppers', 'extra cheese')
```

```
def build_profile(first, last, **user_info):
"""创建一个字典，其中包含我们知道的有关用户的一切"""
profile = {}
❶ profile['first_name'] = first
profile['last_name'] = last ❷ for key, value in user_info.items():
profile[key] = value
return profile
user_profile = build_profile('albert', 'einstein',
location='princeton',
field='physics')
print(user_profile)
```

代码行import pizza 让Python打开文件pizza.py，并将其中的所有函数都复制到这个程序中

### 类

 from collections import OrderedDict

，它兼具列表和字典的主要优点（在将信息关联起来的同时保留原来的顺序）

类名应采用驼峰命名法

```
ord(‘a’)

chr(‘65’)=a
```

###  文件和异常

#### 文件

关键字with 在不再需要访问文件后将其关闭

read()读取文件，读取文件的全部内容

为read() 到达文件末尾时返回一个空字符串，而将这个空字符串显示出来时就是一 个空行

```python
with open('pi_digits.txt') as file_object:
contents = file_object.read()
print(contents.rstrip())
```

逐行进行读取

```python
filename='test.txt'
with open(filename) as f:
    for line in f:
        print(line,end='')
```

为何会出现这些空白行呢？因为在这个文件中，每行的末尾都有一个看不见的换行符，而print 语句也会加上一个换行符，因此每行末尾都有两个换行符：一个来自文件，另一 个来自print 语句。要消除这些多余的空白行，可在print 语句中使用rstrip() ：

```python
filename='test.txt'
with open(filename) as f:
    lines=f.readlines()
print(lines)                              #['afsdaf\n', 'fsadfsadf\n', 'asdfsdaf'] 换行符也会读取进去
```

open中传入另一个参数

‘w’覆盖写文件，‘a’追加写文件

#### 异常

```
try:
		answer = int(first_number) / int(second_number) 
except ZeroDivisionError:
		print("You can't divide by 0!") 
else:
		print(answer)
```

#### 存储数据

```python
import json
numbers = [2, 3, 5, 7, 11, 13]
filename = 'numbers.json' 
with open(filename, 'w') as f_obj: 
    	json.dump(numbers, f_obj)
```

```python
import json
filename = 'numbers.json' 
with open(filename) as f_obj: 
		numbers = json.load(f_obj)
		print(numbers)

```

