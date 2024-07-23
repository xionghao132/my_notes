[TOC]

# Latex公式

## 概述

​	 LaTeX，作为广义上的计算机标记语言（比如HTML），它继承了计算机语言的光荣传统，通过一些简单的代码表达出精确的含义，具有不二义性。其文章排版的结果可以完全按照你的想法来，不仅解决了玄学问题，渲染出来的文章优美；同时，其还可以通过简单的语法写出优雅高贵的数学公式，目前Markdown也已经支持LaTeX语法的公式。

## 语法

### 基础

* 行内公式使用\$…\$

$ f(x) = a+b $

* 行间公式使用\$\$…\$\$。

$$
f(x)=a+b \tag{1}
$$
==注意：==公式中的空格都会被忽略，要使用\quad或者\quad

* 自动编号

```latex
\tag{n}
```

### 常用希腊字母

| 小写命令 | 小写显示 |
| :------: | :------: |
|  \alpha  | $\alpha$ |
|  \beta   | $\beta$  |
|  \gamma  | $\gamma$ |
|  \delta  | $\delta$ |
|  \zeta   |    ζ     |
|  \iota   | $\iota$  |
|  \kappa  | $\kappa$ |
| \lambda  |    λ     |
|   \mu    |  $\mu$   |
|   \rho   |  $\rho$  |
|  \sigma  | $\sigma$ |
|   \tau   |  $\tau$  |
|  \omega  | $\omega$ |

**Tips**

 	如果使用大写的希腊字母，把命令的首字母变成大写即可，例如 \Gamma 输出的是$\Gamma$。

​	 如果使用斜体大写希腊字母，再在大写希腊字母的LaTeX命令前加上var，例如\varGamma 生成 $\varGamma$。

```latex
$$
 \varGamma(x) = \frac{\int_{\alpha}^{\beta} g(t)(x-t)^2\text{ d}t }{\phi(x)\sum_{i=0}^{N-1} \omega_i} \tag{2}
$$
```


$$
\varGamma(x) = \frac{\int_{\alpha}^{\beta} g(t)(x-t)^2\text{ d}t }{\phi(x)\sum_{i=0}^{N-1} \omega_i} \tag{2}
$$

### **常用求和符号和积分号**

|     **命令**      |    **显示结果**     |
| :---------------: | :-----------------: |
|       \sum        |         $∑$         |
|       \int        |         $∫$         |
|  \sum_{i=1}^{N}   |  $\sum_{i=1}^{N}$   |
|   \int_{a}^{b}    |   $\int_{a}^{b}$    |
|       \prod       |       $\prod$       |
|       \iint       |       $\iint$       |
|  \prod_{i=1}^{N}  |  $\prod_{i=1}^{N}$  |
|   \iint_{a}^{b}   |   $\iint_{a}^{b}$   |
|      \bigcup      |      $\bigcup$      |
|      \bigcap      |      $\bigcap$      |
| \bigcup_{i=1}^{N} | $\bigcup_{i=1}^{N}$ |
| \bigcap_{i=1}^{N} | $\bigcap_{i=1}^{N}$ |

* _ 用于下标
* ^ 用于上标

### 其他常用符号

|    **命令**    |   **显示结果**   |
| :------------: | :--------------: |
|  \sqrt[3]{2}   |  $\sqrt[3]{2}$   |
|    \sqrt{2}    |    $\sqrt{2}$    |
|     x_{3}      |     $x_{3}$      |
| \lim_{x \to 0} | $\lim_{x \to 0}$ |
|  \frac{1}{2}   |  $\frac{1}{2}$   |

## 矩阵

```latex
$$\begin{matrix}
…
\end{matrix}$$
```

> 矩阵命令中每一行以 \ 结束，矩阵的元素之间用&来分隔开。

### 简单矩阵

```latex
$$
  \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix} \tag{1}
$$
```
$$
\begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix} \tag{3}
$$

### 带括号的矩阵 \left .. \right

> 想使用大括号需要转义

```latex
$$
\left \{
\begin{matrix}
1&2&3\\
4&5&6\\
7&8&9
\end{matrix}
\right \} \tag{2}
$$
```

$$
\left \{
\begin{matrix}
1&2&3\\
4&5&6\\
7&8&9
\end{matrix}
\right \} \tag{3}
$$

> 方括号不需要转义

```latex
$$
 \left[
 \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix}
  \right] \tag{2}
$$
```

$$
\left[
 \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix}
  \right] \tag{2}
$$

### 带括号的矩阵vmatrix、Vmatrix

```latex
$$
 \begin{vmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{vmatrix} \tag{5}
$$
```

$$
 \begin{vmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{vmatrix} \tag{5}
$$

```latex
$$
 \begin{Vmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{Vmatrix} \tag{5}
$$
```

$$
\begin{Vmatrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{Vmatrix} \tag{5}
$$

### 带省略号的矩阵

> 如果矩阵元素太多，可以使用\cdots ⋯ \ddots ⋱ \vdots ⋮ 等省略符号来定义矩阵。

```latex
$$
\left[
\begin{matrix}
 1      & 2      & \cdots & 4      \\
 7      & 6      & \cdots & 5      \\
 \vdots & \vdots & \ddots & \vdots \\
 8      & 9      & \cdots & 0      \\
\end{matrix}
\right]
$$
```

$$
\left[
\begin{matrix}
 1      & 2      & \cdots & 4      \\
 7      & 6      & \cdots & 5      \\
 \vdots & \vdots & \ddots & \vdots \\
 8      & 9      & \cdots & 0      \\
\end{matrix}
\right]
$$

### 带参数的矩阵

```latex
$$ 
 \left[ \begin{array}{cc|c} 
1 & 2 & 3 \\
4 & 5 & 6 
\end{array} \right] \tag{7} 
$$
```

$$
\left[ \begin{array}{cc|c} 
1 & 2 & 3 \\
4 & 5 & 6 
\end{array} \right] \tag{7}
$$

## 在线编辑latex网站

[latex在线编辑]: latexlive.com

## 集合符号

[(1) Markdown Katex 集合相关符号_COCO56（徐可可）的博客-CSDN博客_markdown集合符号](https://blog.csdn.net/coco56/article/details/100112006)

