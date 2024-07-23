# java自定义排序方法

## Arrays.sort

> ​	一般来说，Arras.sort()方法对数组数据元素进行升序排列，传入自定义比较器，才能进行自定义排序。
>

* 升序

```java
//使用默认的方法
Arrays.sort(arr)

//使用lambda表达式
Arrays.sort(arr,(a,b)->a-b);
```

* 降序

```java
将int->Integer
//使用lambda表达式
Arrays.sort(arr,(a,b)->b-a);
```

更一般的规律：

* return 0:不交换位置，不排序
* return 1:交换位置
* return -1:不交换位置

## Collections.sort

> 方法与上文类似，当时面向的对象是集合对象

==注意：==集合里面都是范型，就没有int,char普通类型自定义的约束