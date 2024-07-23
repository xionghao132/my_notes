# 回溯法DFS
## 1. n个数中找m个数组合

```java
public static void dfs(int[] arr,int num,int start){  //相当于实现找到num个数的组合
        if(list.size()==num)
        {
            res.add(new ArrayList(list));
            return;
        }
        for(int i=start;i<arr.length;i++)
        {
            list.add(arr[i]);
            dfs(arr,num,i+1);
            list.remove(list.size()-1);
        }
    }
```

## 2. 子集

### 2. 1 多一层循环

```java
package com;

import java.util.ArrayList;

public class Solution {
    public static ArrayList<ArrayList<Integer>> res=new ArrayList<>();
    public static ArrayList<Integer> list=new ArrayList<>();
    public static void main(String[] args){
        int[] arr={3,5,7,8};
        for(int i=0;i<=arr.length;i++)
            dfs(arr,i,0);
        System.out.println(res);
    }
    public static void dfs(int[] arr,int num,int start){  //相当于实现找到num个数的组合
        if(list.size()==num)
        {
            res.add(new ArrayList(list));
            return;
        }
        for(int i=start;i<arr.length;i++)
        {
            list.add(arr[i]);
            dfs(arr,num,i+1);
            list.remove(list.size()-1);
        }
    }
}

```

### 2.2直接添加

```java
public static void dfs(int[] arr,int start){  //相当于实现找到num个数的组合
    res.add(new ArrayList(list));
    for(int i=start;i<arr.length;i++)
    {
        list.add(arr[i]);
        dfs(arr,i+1);
        list.remove(list.size()-1);
    }
}
```

## 3.有重复元素

写在`for`语句中

```java
if(i>0&&arr[i]==arr[i-1]) continue;
```

## 4.排列

### 4.1contains实现

```java
package com;

import java.util.ArrayList;

public class Solution {
    public static ArrayList<ArrayList<Integer>> res=new ArrayList<>();
    public static ArrayList<Integer> list=new ArrayList<>();
    public static void main(String[] args){
        int[] arr={1,2,3};
        dfs(arr);
        System.out.println(res);
    }
    public static void dfs(int[] arr){
        if(arr.length==list.size())
        {
            res.add(new ArrayList<>(list));
            return;
        }
        for(int i=0;i<arr.length;i++)
        {
            if(!list.contains(arr[i]))
            {
                list.add(arr[i]);
                dfs(arr);
                list.remove(list.size()-1);
            }
        }
    }
}
```

> ==注意：==使用`contains`可能会很慢，好一点就用visited数组记录

[(4条消息) 算法学习——求有重复元素的全排列（递归）_luladuck的博客-CSDN博客_有重复全排列](https://blog.csdn.net/luladuck/article/details/115414510)

### 4.2交换实现

```java
public class Main {
	static ArrayList<ArrayList<Integer>> res;
    public static void main(String[] args) {
        int[] a= {1,2,3};
        res=new ArrayList<>();
        perm(a,0);
        System.out.println(res.size());
    }
    public static void perm(int[] arr,int begin) {
    	if(begin==arr.length)
    	{
    		ArrayList<Integer> list=new ArrayList<>();
    		for(int i=0;i<arr.length;i++) list.add(arr[i]);
    		res.add(list);
    	}
    	for(int i=begin;i<arr.length;i++)
    	{
    		swap(arr, begin,i);
    		perm(arr,begin+1);
    		swap(arr, begin,i );
    	}
    	
    		
    }
    public static void swap(int[] arr,int i,int j) {
    	if(i!=j)
    	{
    		int t=arr[i];
    		arr[i]=arr[j];
    		arr[j]=t;
    	}
    }
}
```

### 4.3去重

> 写在`for`循环中即可

```java
boolean finish(char list[],int k,int i)
{//第i个元素是否在前面元素[k...i-1]中出现过
	if(i>k)
	{
		for(int j=k;j<i;j++)
			if(list[j]==list[i])
				return false;
	}
	return true;
}
```

