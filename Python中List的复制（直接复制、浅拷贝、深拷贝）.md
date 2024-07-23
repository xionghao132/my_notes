# Python中List的复制（直接复制、浅拷贝、深拷贝）

**如果用 = 直接赋值，是非拷贝方法。**

**这两个列表是等价的，\**修改其中任何一个列表都会影响到另一个列表。\****



## [浅拷贝](https://so.csdn.net/so/search?q=浅拷贝&spm=1001.2101.3001.7020)：

### **copy()**方法

**对于List来说，其第一层，是实现了深拷贝，但对于其内嵌套的List，仍然是浅拷贝。**

**因为\**嵌套的List保存的是地址，复制过去的时候是把地址复制过去了\**，嵌套的List在内存中指向的还是同一个。**

```python
new=old.copy()
```



### 使用列表生成式

**使用列表生成式产生新列表也是一个浅拷贝方法，\**只对第一层实现深拷贝\**。**



### for循环遍历

**通过for循环遍历，将元素一个个添加到新列表中。这也是一个浅拷贝方法，只对第一层实现深拷贝。**



### **使用切片**

**通过使用 [ : ] 切片，可以浅拷贝整个列表，同样的，只对第一层实现深拷贝。**

```python
new=old[:]
```



## **深拷贝：**

**如果用\**deepcopy()\**方法，则无论多少层，无论怎样的形式，得到的新列表都是和原来无关的，这是最安全最清爽最有效的方法。**

```python
import copy
new=copy.deepcopy(old)
```



### clone()

```python
a=b.clone()
```



### contiguous() 

如果想要**断开**这两个**变量之间的依赖**（x本身是contiguous的），就要**使用contiguous()针对x进行变化**，**感觉上就是我们认为的深拷贝**。

 当**调用contiguous()时**，**会强制拷贝一份tensor**，让它的布局和从头创建的一模一样，**但是两个tensor完全没有联系**。

```python

x = torch.randn(3, 2)
y = torch.transpose(x, 0, 1).contiguous()
```





[Python中List的复制（直接复制、浅拷贝、深拷贝）_如何把一个list复制到另一个list中python-CSDN博客](https://blog.csdn.net/qq_24502469/article/details/104185122)



这个链接详细描述了对应的python变量存储过程，从内部存储的角度来解释深拷贝和浅拷贝。

[深入理解Python深拷贝(deepcopy)、浅拷贝（copy）、等号拷贝----看了还不懂找我-CSDN博客](https://blog.csdn.net/corner2030/article/details/126891322)