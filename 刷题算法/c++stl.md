# c++ STL

## vector

  在 c++ 中，vector 是一个十分有用的容器。它能够像容器一样存放各种类型的对象，简单地说，vector是一个能够存放任意类型的动态数组，能够增加和压缩数据。

### 容器特性

1. 顺序序列

   顺序容器中的元素按照严格的线性顺序排序。可以通过元素在序列中的位置访问对应的元素。
2. 动态数组

   支持对序列中的任意元素进行快速直接访问，甚至可以通过指针算述进行该操作。操供了在序列末尾相对快速地添加/删除元素的操作。
3. 能够感知内存分配器的（Allocator-aware）

   容器使用一个内存分配器对象来动态地处理它的存储需求。

### 排序

> 引入头文件 #include `<algorithm>`

* `sort(vec.begin(),vec.end())`
* `sort(vec.rbegin(),vec.rend())`
* `reverse(vec.begin(),vec.end())`

### 基本函数实现

> 引入头文件 #include `<vector>`

1. **构造函数**

   * `vector<int> vec:`创建一个空vector
   * `vector(int nSize)`:创建一个vector,元素个数为nSize
   * `vector(int nSize,const t& t)`:创建一个vector，元素个数为nSize,且值均为t
   * `vector(begin,end)`:复制[begin,end)区间内另一个数组的元素到vector中
2. **增加函数**

   * `void push_back(const T& x)`:向量尾部增加一个元素X
   * `iterator insert(iterator it,const T& x)`:向量中迭代器指向元素前增加一个元素x
3. **删除函数**

   * `void pop_back()`:删除向量中最后一个元素
   * `void clear()`:清空向量中所有元素
   * `iterator erase(iterator it)`:删除向量中迭代器指向元素
   * `iterator erase(iterator first,iterator last)`:删除向量中[first,last)中元素
4. **遍历函数**

   * `iterator begin()`:返回向量头指针，指向第一个元素
   * `iterator end()`:返回向量尾指针，指向向量最后一个元素的下一个位置
   * `for(int i=0;i<vec.size();i++)`
   * `for(int i:ver)`

5.**判断函数**

    *`bool empty() const`:判断向量是否为空，若为空，则向量中无元素

6.**大小函数**

    *`int size() const`:返回向量中元素的个数
    * `int capacity() const`:返回当前向量所能容纳的最大元素值
    * `int max_size() const`:返回最大可允许的 vector 元素数量值

7.**其他类型函数**

    *`void swap(vector&)`:交换两个同类型向量的数据

### 代码

```c++
#include <iostream>
#include <vector>
#include "algorithm"
using namespace  std;
int main() {
    vector<int> vec;
    for(int i=0;i<10;i++)
        vec.push_back(i);
    vec.insert(vec.begin(),11);
    vec.erase(vec.end()-2);
    for(int i=0;i<vec.size();i++) cout<<vec[i]<<" ";
    cout<<endl;
    sort(vec.begin(),vec.end());    //升序
    //sort(vec.rbegin(),vec.rend());  //降序
    for(vector<int>::iterator it=vec.begin();it!=vec.end();it++) cout<<*it<<" ";
    cout<<endl;
    return 0;
}

```

## stack

> 引入头文件 include `<stack>`

1. 主要的操作:

* empty
* size
* top
* push
* pop

2. 代码

```c++
int main() {
    stack<int> st;
    for(int i=0;i<5;i++) st.push(i);
    cout<<st.top();
    while(!st.empty()) st.pop();
    cout<<st.size();
    return 0;
}
```

## queue

> 引入头文件include `<queue>`

1.主要操作

* `front()`：返回 queue 中第一个元素的引用。如果 queue 是常量，就返回一个常引用；如果 queue 为空，返回值是未定义的。
* `back()`：返回 queue 中最后一个元素的引用。如果 queue 是常量，就返回一个常引用；如果 queue 为空，返回值是未定义的。
* `push(const T& obj)`：在 queue 的尾部添加一个元素的副本。这是通过调用底层容器的成员函数 push_back() 来完成的。
* `pop()`：删除 queue 中的第一个元素。
* `size()`：返回 queue 中元素的个数。
* `empty()`：如果 queue 中没有元素的话，返回 true。

2.代码

```c++
int main() {
    queue<int>q;
    for(int i=0;i<5;i++) q.push(i);
    cout<<q.front()<<endl;
    cout<<q.back()<<endl;
    while(!q.empty()) q.pop();
    cout<<q.size()<<endl;
    return 0;
}
```

## set

> 引入头文件include `<set>`

1.主要操作

* `insert(t)`:向集合中插入元素t。
* `erase()`:删除 set 容器中存储的元素。
* `empty()`:若容器为空，则返回 true；否则 false。
* `size()`:	返回当前 set 容器中存有元素的个数。
* `find(val)`:在 set 容器中查找值为 val 的元素，如果成功找到，则返回指向该元素的双向迭代器；反之，则返回和 end() 方法一样的迭代器。另外，如果 set 容器用 const 限定，则该方法返回的是 const 类型的双向迭代器。

2.代码

```c++
int main() {
    set<int> s;
    for(int i=0;i<5;i++) s.insert(i);
    for(auto it=s.begin();it!=s.end();it++) cout<< *it<<" ";
    auto pos=s.find(4);
    cout<<*pos;
    while(!s.empty())
        s.erase(s.begin());
    cout<<s.size()<<endl;
    return 0;
}
```

## map

> 引入头文件include`<map>`

1.主要操作

* `count()`:返回指定元素出现的次数, (帮助评论区理解： 因为key值不会重复，所以只能是1 or 0。
* `insert() `:向map集合中插入元素。
* `empty()`:判断是否为空。
* `erase()`:删除某个位置的元素。
* `size()`:返回键值对个数。
* `find()`:返回满足对应键的位置

2.代码

```c++
int main() {
    map<int,string> map;
    map.insert(pair<int,string>(1,"sfd"));
    map.insert(pair<int,string>(2,"sdf"));
    map[3]="sdfsd";  //第二种插入方式
    auto it=map.find(1);
    if(it!=map.end()) cout<<it->second;
    map.erase(it);
    cout<<map.size()<<endl;
    for(auto a=map.begin();a!=map.end();a++) cout<<a->second<<" ";
    return 0;
}
```
