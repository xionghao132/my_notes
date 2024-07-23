# nohup和tmux

## nohup

### 概述

**nohup** 英文全称 `no hang up`（不挂起），用于在系统后台不挂断地运行命令，退出终端不会影响程序的运行。

**nohup** 命令，在默认情况下（非重定向时），会输出一个名叫 `nohup.out` 的文件到当前目录下，如果当前目录的 `nohup.out` 文件不可写，输出重定向到 **$HOME/nohup.out** 文件中。

**注意：**我们使用ssh连接进行程序运行的时候，如果断开程序也是会退出的。



### 语法格式

```sh
 nohup Command [ Arg … ] [　& ]
```

- **Command**：要执行的命令。
- **Arg**：一些参数，可以指定输出文件。
- **&**：让命令在后台执行，终端退出后命令仍旧执行。



**注意：**我们使用

```
nohup python -u
```

的时候,-u 代表程序不启用缓存，也就是把输出直接放到log中，没这个参数的话，log文件的生成会有延迟



### 重定向

以下命令在后台执行` root `目录下的 `runoob.sh` 脚本，并重定向输入到 `runoob.log `文件：

```
nohup /root/runoob.sh > runoob.log 2>&1 &
```

**2>&1** 解释：

将标准错误 2 重定向到标准输出 &1 ，标准输出 &1 再被重定向输入到 `runoob.log `文件中。

- 0 – `stdin` (standard input，标准输入)
- 1 – `stdout` (standard output，标准输出)
- 2 – `stderr` (standard error，标准错误输出)



### 停止运行

如果要停止运行，你需要使用以下命令查找到 `nohup` 运行脚本到 `PID`，然后使用 kill 命令来删除：

```sh
ps -aux | grep "runoob.sh" 
```

- **a** : 显示所有程序
- **u** : 以用户为主的格式来显示
- **x** : 显示所有程序，不区分终端机

找到 `PID` 后，就可以使用 `kill PID `来删除。

```sh
kill -9  进程号PID
```



## tmux



### 概述

最近使用nohup进行DDP训练，结果不知道是torch版本问题导致不稳定还是nohup训练不稳定，所以看到了tmux这个指令，像发现了新大陆一样。



命令行的典型使用方式是，打开一个终端窗口（terminal window，以下简称"窗口"），在里面输入命令。**用户与计算机的这种临时的交互，称为一次"会话"（session）** 。

会话的一个重要特点是，窗口与其中启动的进程是连在一起的。打开窗口，会话开始；关闭窗口，会话结束，会话内部的进程也会随之终止，不管有没有运行完。

一个典型的例子就是，SSH 登录远程计算机，打开一个远程窗口执行命令。这时，网络突然断线，再次登录的时候，是找不回上一次执行的命令的。因为上一次 SSH 会话已经终止了，里面的进程也随之消失了。

为了解决这个问题，会话与窗口可以"解绑"：窗口关闭时，会话并不终止，而是继续运行，等到以后需要的时候，再让会话"绑定"其他窗口。



### 基本使用

* 启动和退出

```bash
tmux  #启动tmux窗口 底部有一个状态栏。状态栏的左侧是窗口信息（编号和名称），右侧是系统信息。
exit
```

* 快捷键

快捷键都是用`Ctrl+b`作为前缀，

下面是一些会话相关的快捷键。

> - `Ctrl+b d`：分离当前会话。
> - `Ctrl+b s`：列出所有会话。
> - `Ctrl+b $`：重命名当前会话。

**注意**：应该先按住`ctrl+b`，放开后再按其他键才生效

### 记录日志

```sh
script -f a.log   #将当前会话打印信息输出到a.log文件中

exit           #结束保存日志
```

目前遇到的问题是开启了这个记录日志，`conda`的环境就会失效。

[linux中运行tmux时，自动保存日志_tmux怎么保存_Liucxx的博客-CSDN博客](https://blog.csdn.net/weixin_43819842/article/details/125441651)



**tee**命令也可以直接记录日志

如果想同时打印到屏幕和文件里，可以这么写：

```sh
ls -l | tee -a lsls.log #-a是在文件后添加，不加就是覆盖
```

如果想把错误输出也同时打印到屏幕和文件，可以这么写：

```sh
ls -l not_find_runoob 2>&1 | tee -a lsls.log
```

其中，`2>&1` 意思就是把标准报错也作为标准输出。

可以在分离的tmux会话中运行脚本，并将stdout和stderr记录到文件中，如下所示：

```sh
tmux new -d 'script.sh |& tee tmux.log'
```

`-d` 参数是 tmux 命令的一个选项，表示在后台创建会话。使用 `-d` 参数可以在创建会话后立即返回到原来的终端，而不会进入新创建的会话。这样可以在后台执行命令或脚本，而不会影响当前终端的使用。

也可以进入tmux会话中，使用tee命令

```
./my_script.sh |& tee /path/to/script_output.log
```



### 会话管理

* 新建会话

第一个启动的 Tmux 窗口，编号是`0`，第二个窗口的编号是`1`，以此类推。这些窗口对应的会话，就是 0 号会话、1 号会话。

使用编号区分会话，不太直观，更好的方法是为会话起名。

```bash
tmux new -s <session-name>
```

* 分离会话

在 Tmux 窗口中，按下`Ctrl+b d`或者输入`tmux detach`命令，就会将当前会话与窗口分离。

```bash
tmux detach
```

上面命令执行后，就会退出当前 Tmux 窗口，但是会话和里面的进程仍然在后台运行。

* 查看会话

```sh
tmux ls
```

* 接入会话

`tmux attach`命令用于重新接入某个已存在的会话。

```bash
# 使用会话编号
tmux attach -t 0

# 使用会话名称
tmux attach -t <session-name>
```

* 杀死会话

```bash
# 使用会话编号
tmux kill-session -t 0

# 使用会话名称
tmux kill-session -t <session-name>
```

* 切换会话

```sh
# 使用会话编号
tmux switch -t 0

# 使用会话名称
tmux switch -t <session-name>
```

* 重命名

```sh
tmux rename-session -t 0 <new-name>
```



### 使用流程

1. 新建会话`tmux new -s my_session`。
2. 在 Tmux 窗口运行所需的程序。
3. 按下快捷键`Ctrl+b d`将会话分离。
4. 下次使用时，重新连接到会话`tmux attach-session -t my_session`。



### 窗格操作

Tmux 可以将窗口分成多个窗格（pane），每个窗格运行不同的命令。以下命令都是在 Tmux 窗口中执行。

* 划分窗格

```bash
# 划分上下两个窗格
tmux split-window

# 划分左右两个窗格
tmux split-window -h
```

* 移动光标

```bash
# 光标切换到上方窗格
tmux select-pane -U

# 光标切换到下方窗格
tmux select-pane -D

# 光标切换到左边窗格
tmux select-pane -L

# 光标切换到右边窗格
tmux select-pane -R
```

* 交换窗格位置

```bash
# 当前窗格上移
tmux swap-pane -U

# 当前窗格下移
tmux swap-pane -D
```

* 窗格快捷键

1. `Ctrl+b %`：划分左右两个窗格。
2. `Ctrl+b "`：划分上下两个窗格。
3. `Ctrl+b <arrow key>`：光标切换到其他窗格。`<arrow key>`是指向要切换到的窗格的方向键，比如切换到下方窗格，就按方向键`↓`。
4. `Ctrl+b ;`：光标切换到上一个窗格。
5. `Ctrl+b o`：光标切换到下一个窗格。
6. `Ctrl+b {`：当前窗格与上一个窗格交换位置。
7. `Ctrl+b }`：当前窗格与下一个窗格交换位置。
8. `Ctrl+b Ctrl+o`：所有窗格向前移动一个位置，第一个窗格变成最后一个窗格。
9. `Ctrl+b Alt+o`：所有窗格向后移动一个位置，最后一个窗格变成第一个窗格。
10. `Ctrl+b x`：关闭当前窗格。
11. `Ctrl+b !`：将当前窗格拆分为一个独立窗口。
12. `Ctrl+b z`：当前窗格全屏显示，再使用一次会变回原来大小。
13. `Ctrl+b Ctrl+<arrow key>`：按箭头方向调整窗格大小。
14. `Ctrl+b q`：显示窗格编号。



### 窗格管理

除了将一个窗口划分成多个窗格，Tmux 也允许新建多个窗口。

* 新建窗口

```bash
tmux new-window

# 新建一个指定名称的窗口
tmux new-window -n <window-name>
```

* 切换窗口

```bash
# 切换到指定编号的窗口
tmux select-window -t <window-number>

# 切换到指定名称的窗口
tmux select-window -t <window-name>
```

* 重命名窗口

```bash
tmux rename-window <new-name>
```

* 窗口快捷键
	 - `Ctrl+b c`：创建一个新窗口，状态栏会显示多个窗口的信息。

	 - `Ctrl+b p`：切换到上一个窗口（按照状态栏上的顺序）。
   - `Ctrl+b n`：切换到下一个窗口。
   - `Ctrl+b <number>`：切换到指定编号的窗口，其中的`<number>`是状态栏上的窗口编号。
   - `Ctrl+b w`：从列表中选择窗口。
   - `Ctrl+b ,`：窗口重命名。



### 其他

```bash
# 列出所有快捷键，及其对应的 Tmux 命令
tmux list-keys

# 列出所有 Tmux 命令及其参数
tmux list-commands

# 列出当前所有 Tmux 会话的信息
tmux info

# 重新加载当前的 Tmux 配置
tmux source-file ~/.tmux.conf
```


[Tmux 使用教程 - 阮一峰的网络日志 (ruanyifeng.com)](https://www.ruanyifeng.com/blog/2019/10/tmux.html)
