# DOS批处理

## 概念

bat脚本就是DOS批处理脚本，就是将一些列DOS命令按照一定顺序排列而形成的集合，运行在windows命令行环境上。

## DOS基本命令介绍

注释**rem** 注释符，也可以用两个冒号代替(::)

```powershell
REM [comment] 
```

**echo** 显示信息，或启用或关闭命令回显。@字符放在命令前将关闭该命令回显，无论此时echo是否为打开状态。

```
echo on 批处理命令在执行时显示自身命令行
echo off 批处理命令在执行时不显示自身命令行
@echo off 关闭echo off命令行自身的显示
echo Hello World 打印Hello World
echo. 输出空行,"."可以用，：；”／[\]＋等任一符号替代
echo test > file.txt 创建包含字符test的file.txt文件
echo y | del d:\temp\*.txt 输入y确认删除
```

**pause** 暂停并输出“请按任意键继续. . .”

```
pause 等待并提示"请按任意键继续. . ."
pause > nul 等待但不出现提示语
echo wait a moment.. & pause > nul 输出指定输出语"wait a moment.."并等待操作
```

**errorlevel。**程序执行结果返回码，执行成功返回0，失败返回为1

**start** 启动一个单独的窗口以运行指定的程序或命令，程序继续向下执行。

```
START [command/program] [parameters]
```

（6）**exit** 退出CMD.EXE程序或当前批处理脚本

```
EXIT [/B] [exitCode]
参数说明：
/B：指定要退出当前批处理脚本而不是 CMD.EXE。如果从一个批处理脚本外执行，则会退出 CMD.EXE
exitCode：指定一个数字号码。如果指定了 /B，将 ERRORLEVEL设成那个数字。如果退出 CMD.EXE，则用那个数字设置过程退出代码
```

（7）**cls** 清除屏幕内容

（8）**help** 提供 Windows 命令的帮助信息

```
HELP [command]
```

## 常用变量

```
%CD%  获取当前目录
%PATH%  获取命令搜索路径
%DATE%  获取当前日期。
%TIME%  获取当前时间。
%RANDOM% 获取 0 和 32767 之间的任意十进制数字。
%ERRORLEVEL% 获取上一命令执行结果码
```



```
set VAR1="I Love BAT Script"
D:\>echo %VAR1%
```



## 字符串操作

字符串截取。使用命令 **echo %var:~n,k%，**其中"%var"，表示待截取字符的字符串，"~"取字符标志符，"n”表示字符截取起始位置，"k" 表示截取结束位置（不包含该字符）。

```powershell
@echo off 
set str=superhero
echo str=%str% 
echo str:~0,5=%str:~0,5%
echo str:~3=%str:~3%
echo str:~-3=%str:~-3% 
echo str:~0,-3=%str:~0,-3% 
pause
```



字符串替换。使用命令**%var:old_str=new_str%**

```powershell
@echo off 
set str=hello world!
set temp=%str:hello=good% 
echo %temp% 
pause 
```



## **文件操作命令**

（1）copy 文件复制命令

```powershell
copy d:\temp1\file1.txt d:\temp2 将文件file1.txt复制到temp2目录，有相同文件提示
copy d:\temp1\file1.txt d:\temp2 /y 将文件file1.txt复制到temp2目录,有相同文件覆盖原文件，不提示
copy d:\temp1\* d:\temp2 /y 将temp1目录下的所有文件复制到temp2目录,有相同文件覆盖原文件，不提示
```

（2）xcopy 目录复制命令

```powershell
xcopy temp1 d:\temp2 /y 将temp1目录下的文件复制到temp2目录，不包括temp1子目录下的文件。
xcopy temp1 d:\temp2 /s /e /y 将temp1目录下的文件复制到temp2目录，包括temp1子目录下的文件
```

（3）type 显示文件内容命令

```powershell
type file1.txt 查看file1文件内容
type file1.txt file2.txt  #查看file1和file2文件内容
type file1.txt > file2.txt  #将file1.txt文件内容重定向到file2.txt
type nul > file1.txt #创建文件
```

（4）ren 重命名文件命令

```powershell
ren d:\temp1\file1.txt file2.txt 修改temp目录下的file1.txt文件名为file2.txt
```

（5）del 删除文件命令

```powershell
del d:\temp1\file1.txt 删除temp目录下的file1.txt文件
del d:\temp\*.txt  删除temp目录下的后缀为.txt的文件
```

## **目录操作命令** 

（1） cd 显示当前目录或切换目录

```
cd d:\temp1 切换到temp1目录，当前目录是d盘
cd /d d:\temp1 切换到temp1目录，当前目录非d盘
cd .. 切换到上一级目录
```

（2）mkdir 创建目录

```
mkdir test 在当前目录下创建test目录
mkdir d:\temp1\test 在temp1目录下创建test目录，如果temp1目录不存在，自动创建
```

（3）rmdir 删除目录

```
rmdir d:\temp1 删除空目录temp1，非空则删除失败
rmdir d:\temp1 /s /q 删除temp1目录，包括子目录(/s)，并且删除时不提示(/q)
```

（4）dir 显示目录下的子目录和文件

```
dir d:\temp1 显示temp1目录下的文件和目录信息，显示信息包含日期、时间、文件类型和文件名
dir d:\temp1 /a:a /b 只显示temp1目录下（不包括子目录）的文件的绝对路径，不显示日期、时间、文件类型和文件名
dir d:\temp1 /b /s /o:n /a:a  显示temp1路径下（包括子目录）的所有文件的绝对路径。输出文件按照文件名数字顺序排序
dir d:\temp1\*.txt /a:a /b /o:n 显示.txt后缀文件，并且按照文件名顺序排序(/on),其他排序方法查看help dir
```

 说明：

 （1）/b表示去除摘要信息，仅显示完整路径、/s表示循环列举文件夹中的内容、/o:n 表示根据文件名排序、/a:a 表示只枚举文件而不枚举其他。

 （2）单独dir /b与dir /s 都不会显示完整路径，只有这两个组合才会显示完整路径。 



## if/else判断句使用

  if/else条件语句，用来判定是否符合规定的条件，从而决定执行不同的命令。 在CMD下使用“IF /?”打开 IF 的系统帮助,IF有3种基本的用法,如下

```
IF [NOT] ERRORLEVEL number command
IF [NOT] string1==string2 command
IF [NOT] EXIST filename command
```

 说明：

```
  NOT：指定只有条件为 false 的情况下，Windows 才应该执行该命令。
  ERRORLEVEL number：如果最后运行的程序返回一个等于或大于指定数字的退出代码，指定条件为 true。
  string1==string2：如果指定的文字字符串匹配，指定条件为 true。
  EXIST filename：如果指定的文件名存在，指定条件为 true。
  command：如果符合条件，指定要执行的命令。如果指定的条件为 FALSE，命令后可跟 ELSE 命令，该命令将在 ELSE 关键字之后执行该命令。
```

  ELSE 子句必须出现在同一行上的 IF 之后。例如:

```
    IF EXIST filename. (
        del filename.
    ) ELSE (
            echo filename. missing.
    )
```

 或者如果都放在同一行上，以下子句有效:

```
IF EXIST filename. (del filename.) ELSE echo filename. missing
```

 如果命令扩展被启用，IF 会如下改变:

```
IF [/I] string1 compare-op string2 command
IF CMDEXTVERSION number command
IF DEFINED variable command
```

 其中， compare-op 可以是:

```
EQU - 等于
NEQ - 不等于
LSS - 小于
LEQ - 小于或等于
GTR - 大于
GEQ - 大于或等于
```

  而 /I 开关(如果指定)说明要进行的字符串比较不分大小写。/I 开关可以用于 IF 的 string1==string2 的形式上。这些比较都是通用的；原因是，如果 string1 和 string2 都是由数字组成的，字符串会被

转换成数字，进行数字比较。

## for语句使用

 cmd命令行窗口下，输入help for 或者 for /? 查看for语句的使用方法。for语句基本格式如下：

 FOR %variable IN (set) DO command [command-parameters]

 参数说明：

```
%variable  指定一个单一字母可替换的参数。注意：批处理脚本中使用%%variable
(set)  指定一个或一组文件。可以使用通配符。
command  指定对每个文件执行的命令。
command-parameters 为特定命令指定参数或命令行开关。
```

 for语句还有4个参数，分别是 /d /r /l /f ，下面分别介绍这4个参数对应的for语句命令。

（1）/D参数的fro语句格式

```
FOR /D %variable IN (set) DO command [command-parameters]
```

 说明：如果（set）集中包含通配符，则指定与目录名匹配，而不与文件名匹配。

 实例：打印C盘根目录下的目录名

```
@echo off
for /d %%i in (c:/*) do (
  echo %%i
)
pause
```

（2）/R参数的fro语句格式

```
FOR /R [[drive:]path] %variable IN (set) DO command [command-parameters] 
```

 说明：递归查询指定目录下的匹配文件。默认使用当前目录。

 实例：打印D盘目录及子目录下的后缀为.txt和.py的文件

```
@echo off
for /r d:/temp %%i in ( *.txt *.py ) do (
  echo %%i
)
pause
```

（3）/L参数的fro语句格式

```
FOR /L %variable IN (start,step,end) DO command [command-parameters]
```

 该集表示以增量形式从开始到结束的一个数字序列。因此，(1,1,5)将产生序列1 2 3 4 5，(5,-1,1)将产生序列(5 4 3 2 1)

实例：打印10以内的奇数

```
@echo off
for /l %%i in (1,2,10) do (
  echo %%i
)
pause
```

（4）FOR语句的/F参数包含如下3种命令格式：

```
  FOR /F ["options"] %variable IN (file-set) DO command [command-parameters]
  FOR /F ["options"] %variable IN ("string") DO command [command-parameters]
  FOR /F ["options"] %variable IN ('command') DO command [command-parameters]
```

 说明：包含/F的参数可以处理文件内容（file-set）、字符串("string")以及执行指定命令('command')返回回的值。可以通过设置["options"]值实现相关需求。["options"]值包含关键字说明如下：

```
eol=c 处理时跳过起始为c字符的行，通常用于跳过注释行。
skip=n  跳过文件开始的n行
delims=xxx  指定分隔符集。这个替换了空格和制表符的默认分隔符集。
tokens=x,y,m-n  被分隔各字段的处理。
usebackq   需使用双引号包含文件名时考虑，具体使用执行help for查看
```

 上面的描述有些地方可能不好裂解，学习并执行完如下几个实例观察输出结果，再去理解效果会更快。

 实例：操作temp.txt文件内容。

```
@echo off

type nul > temp.txt
echo ;Test for /f parameter >> temp.txt
echo line1 1 2 3 >> temp.txt
echo line2 1 2 3 >> temp.txt
echo line3 1 2 3 >> temp.txt
echo 11 12 13 14 15 16 >> temp.txt
echo 21,22,23,24,25,26 >> temp.txt
echo 31-32-33-34-35-36 >> temp.txt
for /F "skip=4 eol=;  tokens=1,3* delims=,- " %%i in (temp.txt) do (
  echo  i=%%i, j=%%j, k=%%k
)
pause
del temp.txt
```

## bat脚本常用实例

据输入选项操作

```
@echo off

set /p var="Please input the number(1,2,3):"

if %var% == 1 (
  echo "the number equal to 1"
) else if %var% == 2 (
  echo "the number equal to 2"
) else if %var% == 3 (
  echo "the number equal to 3"
) else (
  echo "input wrong number,exit program."
)

pause
```

