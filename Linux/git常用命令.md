[TOC]

# git常用命令

## git命令框简介

 	模拟linux命令窗口

## 初始化

 	下载好git后配置环境参数，用户名和邮箱

> git在敲代码的时候可以用tab补全

```
git clone http://...  //复制对应网址
git config --list    //显示所有设置参数
```

![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170033.png)

![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170200.png)

> 在该文件夹下使用命令  

```
git init   //初始化 获得.git的版本库
```

## 配置代理

经常我们要访问外网的东西，例如`github`和`hugging face`时候，需要设置本地电脑的代理。

```sh
 git config --global https.proxy http://127.0.0.1:7890
 git config --global http.proxy http://127.0.0.1:7890
```

https://blog.csdn.net/u014723479/article/details/130716844

## 本地仓库操作

### 常用代码

先进入repo2文件夹

```
git status      //查看当前所有文件的状态
git status -s   //显示的内容更加简洁
git add 文件名   //添加到暂存区
git reset HEAD 文件名  //将已经被跟踪的文件变成未被跟踪的文件
git commit -m "string"  // string是打印到日志中的  提交所有文件
git rm 文件名      //删除文件 rm直接进入暂存区 然后提交才算删除
```

 	==要想提交文件必须先放到暂存区==
 	
 	使用右键删除的话要先加载到暂存区，在提交才行。

### 将文件加入忽略列表

 	希望一些文件不被git管理 例如日志文件还有中间编译产生的文件.class
 	在工作目录创建一个.gitignore文件 —>用命令窗口实现  

`忽略的格式如下：`

```
*.a        //所有以.a结尾文件忽略
!lib.a     //除了lib.a文件
/todo      //当前目录todo
build/     //build目录下的文件
doc/.*txt   //doc文件下面的所有txt文件
doc/**/*.pdf  //doc下包含子目录下所有pdf结尾的目录
```

### 查看日志文件

```
git log   //查看日志
```

 	敲回车可以不断地查看日志（end）结束标志。<font color='cornflowerblue'>q</font>来退出界面

## 远程仓库操作

### 常用命令

```
git remote       //查看与哪个远程仓库建立了关联  clone 默认配置好的是origin
git remote -v    //显示与哪个仓库建立了联系
git remote show origin(仓库名)  //显示更加详细 与分支有关
```

### 添加仓库

```
git remote add origin(可以自己选) http:// 远程仓库url
git clone url  //在仓库文件夹等同目录开始克隆
```

![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170219.png)

 	<font color='orange'>在该目录下打开命令窗口</font> 

### 移除仓库

 	只是将远程仓库的关联去掉，但是云端不会受到影响

```sh
git remote rm origin
```

### 抓取和拉取文件

```
git fetch           //从远程仓库抓取最新数据 不会自动出现
git merge origin/master    //手动合并到工作区，出现到工作区
git pull origin master     //拉取master分支 等同于 fetch和merge两个操作
```

 	==注意：==如果当前仓库不是从远程仓库克隆的，而是本地创建的仓库，并且仓库中存在文件，我们再从远程仓库拉取文件的时候会报错（fatal refusing to merge unrelated histories）
 	
 	==例如：==自己创建了一个hello.txt添加并且提交到了本地仓库，然后又从远程仓库拉取，就会报错。
 	
 	==解决：==添加参数

```
git pull origin master --allow-unrelated-histores //命令框会进入日志编辑 i 插入 wq保存并且退出
```

### 推送文件

```
git push origin master        //推送到远程仓库 开始认证
git commit -a -m "string"                //直接加到暂存区然后提交
```

## git分支

### 本地仓库分支

<font color='cornflowerblue'>*</font>表示当前处于哪个分支下面

```
git branch          //查看本地分支
git branch b1(分支的名字)  //创建b1分支 
git checkout b1      //切换到b1分支
```



### 远程仓库分支

```
git branch -r      //查看远程分支
git branch -a       //查看所有分支 包括本地和远程
git push origin b1   //将b1分支推送到远程仓库
```

### 合并分支

 	在master合并b1分支，应当在master分支使用下面命令

```
git merge b1       //合并分支 
```

 	当不同分支文件发生冲突 需要手动删除文件中的冲突，然后使用如下命令

```
git add 文件名             //表明自己已经修改好了冲突
git commit -m "string"    //提交
```

### 删除分支

```
git branch -d b2       //删除本地分支 远程分支依旧存在
git branch -D b2       //如果该分支进行了开发，-d无法删除 强制删除使用-D
git push origin -d b2   //删除远程分支b2
```

## git标签

 	一般都在版本交替的时间段打标签

### 列出标签

```
git tag v0.1(tagName)     //新建一个tag
git tag               //查看标签
git show  v0.1                   //查看tag信息
```

### 推送标签

```
git push origin v0.1    //将标签推送到远程仓库
```

### 检出标签

```
git checkout -b b3 v1.0         //创建一个新的分支b3指向标签v1.0 进行升级开发
```

### 删除标签

```
 git tag -d v0.2               //删除本地标签
 git push origin :refs/tags/v1.0  //删除远程仓库标签v1.0 注意冒号前面有空格
```



------

# git-lfs

需要先进行安装git-lfs软件



初始化

```sh
git lfs intall

git clone xx.git  #大小文件都下载，但是没有进度

git lfs clone xx.git   #大小文件都有进度应该
```



只下载小文件

```sh
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/bigscience/bloom-7b1

```

下载文件

```sh
git lfs pull --include="*.bin"
```



选择想要追踪的文件

```sh
#选择要用 LFS 追踪的文件：

git lfs track "*.svg"
# 或者具体到某个文件
git lfs track "2.png"
git lfs track "example.lfs"

#不跟踪文件
git lfs untrack
```





# tortoiseGit图形化界面

 	学习了git的常用命令，使用图形化界面就显得更加容易。

## 创建和克隆仓库

> ==git create repository here== 创建仓库

​																				<img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170223.png" style="zoom:50%;" />		  

>   复制对应远程仓库即可克隆

​                                              					      <img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170224.png" style="zoom:50%;" />

> 添加到暂存区

​							                                                    <img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170225.png" style="zoom:50%;" />

> ​	点击文件，右键 ==git commit==,填写message(日志)，即可提交到本地仓库

​                														  <img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170226.png" style="zoom:50%;" />  

> 也可以不添加文件直接提交

<img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170227.png" style="zoom:50%;" />

## 推送本地仓库到远程仓库

> 选择一些推送==分支== local是本地分支  remote是要推送到目的地的远程分支 <font color='red'>最好对应</font>

<img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170228.png" style="zoom:50%;" />

 	<font color='cornflowerblue'>如果自己创建的仓库，要点击manage进行设置，local:origin remote:url (远程仓库)</font>

## 拉取、创建、切换、合并分支

> 拉取分支

<img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170230.png" style="zoom:50%;" />

> 创建分支

<img src="https://gitee.com/HB_XN/picture/raw/master/img/20210428170229.png" style="zoom:50%;" />

> 点击上图中的==switch/checkout==切换分支

> 点击上图的==merge==合并分支

# 启动ssh协议传输数据

## 回顾

 	前面我们所使用的协议是https协议进行数据传输，绑定一次后，就会将账户密码永久保存在我们的电脑上面。

![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170231.png)

## ssh绑定

 	使用ssh通信时，推荐基于密钥的验证方式。在命令输入框输入如下代码创建一对密钥，并且把公钥放在我们的服务器上面。

```
ssh-kengen -t rsa
```

![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170232.png)

> 在.ssh文件夹中会生成公钥和私钥 

> 将id_rsa.pub中的内容复制到服务器当中即可完成绑定