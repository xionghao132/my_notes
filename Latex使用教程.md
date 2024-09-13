# Latex使用教程



## 概述

latex和word一样都是排版工具，word编辑起来很方便，但是调整各种格式很难，latex非所见所得，需要进行编译才能展示出来，优点就是格式都自动排版好了，缺点就是上手难度高一点，熟练之后直接起飞。



## 开始

`\documentclass` 可以修改字体大小，纸张大小，单面打印以及文件类型。

`\begin{documnet}` `\end{document}`主要是来包裹正文的。

`\usepackage{}`导入包，也可以加载宏宝的时候设置参数。

```latex
\documentclass[12pt, a4paper, oneside]{ctexart}
\usepackage{amsmath, amsthm, amssymb, graphicx}
\usepackage[bookmarks=true, colorlinks, citecolor=blue, linkcolor=black]{hyperref}

% 导言区

\title{我的第一个\LaTeX 文档}
\author{Dylaaan}
\date{\today}

\begin{document}  

\maketitle

这里是正文. 

\end{document}
```



## 正文

正文是直接在`document`中进行填写，没有必要加入空格和缩进，文档默认进行首行缩进，相邻的两行会当成一行，所以要分段的话就要再敲一行，这样保证文档不会多的空行和缩进。

`\newpage`另起一页

在正文中，还可以设置局部的特殊字体：

| 字体     | 命令      |
| -------- | --------- |
| 直立     | \textup{} |
| 意大利   | \textit{} |
| 倾斜     | \textsl{} |
| 小型大写 | \textsc{} |
| 加宽加粗 | \textbf{} |



## 章节

`\section` 一级标题

`\subsection` 二级标题



## 目录

在有了章节的结构之后，使用`\tableofcontents`命令就可以在指定位置生成目录。通常带有目录的文件需要编译两次，因为需要先在目录中生成`.toc`文件，再据此生成目录。



## 图片

插入图片需要使用`graphicx`宏包，建议使用如下方式：

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=8cm]{图片.jpg}
    \caption{图片标题}
\end{figure}
```

其中，`[htbp]`的作用是自动选择插入图片的最优位置，`\centering`设置让图片居中，`[width=8cm]`设置了图片的宽度为8cm，`\caption{}`用于设置图片的标题。



```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=8cm]{图片.jpg}
    \caption{图片标题}
\end{figure*}
```

1. figure\* 中 加\*表示占用双栏，不加就是单栏。**这对于表格和公式同样适用**。
2. 图形的放置位置，这一可选参数项可以是下列字母的任意组合。设置相应的参数就对应相对的位置，如果多个参数 如 `\begin{figure*}[htbp]`，它会按照设置选择合适的位置调整，建议设置为  `htbp`，除非排版特别要求。

- `h` 当前位置。 将图形放置在 正文文本中给出该图形环境的地方。如果本页所剩的页面不够， 这一参数将不起作用
- `t` 顶部。 将图形放置在页面的顶部
- `b` 底部。 将图形放置在页面的底部
- `p`浮动页。将图形放置在一只允许 有浮动对象的页面上

## 表格

LaTeX中表格的插入较为麻烦，可以直接使用[Create LaTeX tables online – TablesGenerator.com](https://link.zhihu.com/?target=https%3A//www.tablesgenerator.com/%23)来生成。建议使用如下方式：

```tex
\begin{table}[htbp]
    \centering
    \caption{表格标题}
    \begin{tabular}{ccc}
        1 & 2 & 3 \\
        4 & 5 & 6 \\
        7 & 8 & 9
    \end{tabular}
\end{table}
```



## 列表

LaTeX中的列表环境包含无序列表`itemize`、有序列表`enumerate`和描述`description`，以`enumerate`为例，用法如下：

```tex
\begin{enumerate}
    \item 这是第一点; 
    \item 这是第二点;
    \item 这是第三点. 
\end{enumerate}
```

另外，也可以自定义`\item`的样式：

```tex
\begin{enumerate}
    \item[(1)] 这是第一点; 
    \item[(2)] 这是第二点;
    \item[(3)] 这是第三点. 
\end{enumerate}
```



## 定理环境

定理环境需要使用`amsthm`宏包，首先在导言区加入：

```text
\newtheorem{theorem}{定理}[section]
```

其中`{theorem}`是环境的名称，`{定理}`设置了该环境显示的名称是“定理”，`[section]`的作用是让`theorem`环境在每个section中单独编号。在正文中，用如下方式来加入一条定理：

```tex
\begin{theorem}[定理名称]
    这里是定理的内容. 
\end{theorem}
```

其中`[定理名称]`不是必须的。另外，我们还可以建立新的环境，如果要让新的环境和`theorem`环境一起计数的话，可以用如下方式：

```tex
\newtheorem{theorem}{定理}[section]
\newtheorem{definition}[theorem]{定义}
\newtheorem{lemma}[theorem]{引理}
\newtheorem{corollary}[theorem]{推论}
\newtheorem{example}[theorem]{例}
\newtheorem{proposition}[theorem]{命题}
```

另外，定理的证明可以直接用`proof`环境。



## 页面

最开始选择文件类型时，我们设置的页面大小是a4paper，除此之外，我们也可以修改页面大小为b5paper等等。

一般情况下，LaTeX默认的页边距很大，为了让每一页显示的内容更多一些，我们可以使用`geometry`宏包，并在导言区加入以下代码：

```tex
\usepackage{geometry}
\geometry{left=2.54cm, right=2.54cm, top=3.18cm, bottom=3.18cm}
```

另外，为了设置行间距，可以使用如下代码：

```tex
\linespread{1.5}
```



## 页码

默认的页码编码方式是阿拉伯数字，用户也可以自己设置为小写罗马数字：

```tex
\pagenumbering{roman}
```

另外，`aiph`表示小写字母，`Aiph`表示大写字母，`Roman`表示大写罗马数字，`arabic`表示默认的阿拉伯数字。如果要设置页码的话，可以用如下代码来设置页码从0开始：

```tex
\setcounter{page}{0}
```



## 数学公式的输入方式

行内公式通常使用`$..$`来输入，这通常被称为公式环境，例如：

```tex
若$a>0$, $b>0$, 则$a+b>0$.
```

公式环境通常使用特殊的字体，并且默认为斜体。需要注意的是，只要是公式，就需要放入公式环境中。如果需要在行内公式中展现出行间公式的效果，可以在前面加入`\displaystyle`，例如

```tex
设$\displaystyle\lim_{n\to\infty}x_n=x$.
```

### 行间公式

行间公式需要用`$$..$$`来输入，笔者习惯的输入方式如下：

```tex
若$a>0$, $b>0$, 则
$$
a+b>0.
$$
```

这种输入方式的一个好处是，这同时也是Markdown的语法。需要注意的是，行间公式也是正文的一部分，需要与正文连贯，并且加入标点符号。

关于具体的输入方式，可以参考[在线LaTeX公式编辑器-编辑器 (latexlive.com)](https://link.zhihu.com/?target=https%3A//www.latexlive.com/)，在这里只列举一些需要注意的。

### 上下标

上标可以用`^`输入，例如`a^n`，下标可以用`_`来输入，例如`a_1` 。上下标只会读取第一个字符，如果上下标的内容较多的话，需要改成`^{}`或`_{}`。

### 分式

分式可以用`\dfrac{}{}`来输入，例如`\dfrac{a}{b}`。为了在行间、分子、分母或者指数上输入较小的分式，可以改用`\frac{}{}`，例如`a^\frac{1}{n}`

### 括号

括号可以直接用`(..)`输入，但是需要注意的是，有时候括号内的内容高度较大，需要改用`\left(..\right)`。例如`\left(1+\dfrac{1}{n}\right)^n`

在中间需要隔开时，可以用`\left(..\middle|..\right)`。

另外，输入大括号{}时需要用`\{..\}`，其中`\`起到了转义作用。

### 加粗

对于加粗的公式，建议使用`bm`宏包，并且用命令`\bm{}`来加粗，这可以保留公式的斜体。

### 大括号

在这里可以使用`cases`环境，可以用于分段函数或者方程组，例如

```tex
$$
f(x)=\begin{cases}
    x, & x>0, \\
    -x, & x\leq 0.
\end{cases}
$$
```



### 多行公式

多行公式通常使用`aligned`环境，例如

```tex
$$
\begin{aligned}
a & =b+c \\
& =d+e
\end{aligned}
$$
```



### 矩阵和行列式

矩阵可以用`bmatrix`环境和`pmatrix`环境，分别为方括号和圆括号，例如

```tex
$$
\begin{bmatrix}
    a & b \\
    c & d
\end{bmatrix}
$$
```

如果要输入行列式的话，可以使用`vmatrix`环境，用法同上。

[【LaTeX】新手教程：从入门到日常使用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/456055339)



## 实战经验



英文使用pdflatex

中文使用xelatex



\newcommand 定义自己的命令

然后使用里面的命令当成占位符即可

``index''   这样可以把index编译使用双引号包裹

引用的时候\cite 放一个人，\citep放一个人

\---表示一条横线

\ref一般是引用的图片

\emph{} 强调斜体



\textit{}设置为斜体





\textbf{}加粗字体

vscode可以自动换行 不然一行太长了

\$\sim$表示波浪 ~

\url{}     链接

usepackage{xurl}  脚注下面的链接会自动换行

这个博主有一些latex内容值得看看

[LaTeX文档插入算法代码_latex插入算法-CSDN博客](https://blog.csdn.net/qq_36158230/article/details/124694787)
