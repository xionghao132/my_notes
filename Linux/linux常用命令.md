[TOC]

# linux常用命令

## 目录操作命令

### 目录切换 

```sh
cd /        切换到根目录
cd /usr        切换到根目录下的usr目录
cd ../        切换到上一级目录 或者  cd ..
cd ~        切换到home目录
cd -        切换到上次访问的目录
```

### 目录查看

```sh
ls                查看当前目录下的所有目录和文件
ls -a            查看当前目录下的所有目录和文件（包括隐藏的文件）
ls -l 或 ll       列表查看当前目录下的所有目录和文件（列表查看，显示更多信息）
ls /dir            查看指定目录下的所有目录和文件   如：ls /usr
```

### 删除目录或者新建文件

```sh
新建目录
mkdir   aa  
删除文件：
rm 文件        删除当前目录下的文件
rm -f 文件    删除当前目录的的文件（不询问）

删除目录：
rm -r aaa    递归删除当前目录下的aaa目录
rm -rf aaa    递归删除当前目录下的aaa目录（不询问）

全部删除：
rm -rf *    将当前目录下的所有目录和文件全部删除
rm -rf /*    【自杀命令！慎用！慎用！慎用！】将根目录下的所有文件全部删除
```

### 目录修改

```sh
mv aaa bbb      //将目录aaa改为bbb
mv /usr/tmp/aaa /usr  //将/usr/tmp目录下的aaa目录剪切到 /usr目录下面
cp /usr/tmp/aaa  /usr   //将/usr/tmp目录下的aaa目录复制到 /usr目录下面 
```

### 搜索目录

```sh
find /usr/tmp -name 'a*'       //查找/usr/tmp目录下的所有以a开头的目录或文件
```

## 文件操作命令

### 新建文件

```sh
touch aa.txt      //在当前目录创建一个名为aa.txt的文件   
```

### 删除文件

```sh
rm -rf 文件名
```

### 修改文件

```sh
i          //进入编辑
esc        //退出编辑模式
q!          //撤销本次修改并退出编辑
wq         //保存并且退出
vi 文件名    //打开文件
```

## 压缩文件操作

### 打包和压缩

```sh
tar -zcvf ab.tar aa.txt bb.txt  //打包和压缩
：z：调用gzip压缩命令进行压缩
  c：打包文件 create
  v：显示运行过程 verbose
  f：指定文件名 filename
  x: 解压使用的 extract
tar -xvf ab.tar -C /usr------C代表指定解压的位置
```

## 链接

Linux ln命令是一个非常重要命令，它的功能是为某一个文件在另外一个位置建立一个同步的链接。

当我们需要在不同的目录，用到相同的文件时，我们不需要在每一个需要的目录下都放一个必须相同的文件，我们只要在某个固定的目录，放上该文件，然后在 其它的目录下用ln命令链接（link）它就可以，不必重复的占用磁盘空间。

Linux文件系统中，有所谓的链接(link)，我们可以将其视为档案的别名，而链接又可分为两种 : 硬链接(hard link)与软链接(symbolic link)，硬链接的意思是一个档案可以有多个名称，而软链接的方式则是产生一个特殊的档案，该档案的内容是指向另一个档案的位置。硬链接是存在同一个文件系统中，而软链接却可以跨越不同的文件系统。

不论是硬链接或软链接都不会将原本的档案复制一份，只会占用非常少量的磁碟空间。

**软链接**：

- 1.软链接，以路径的形式存在。类似于Windows操作系统中的快捷方式
- 2.软链接可以 跨文件系统 ，硬链接不可以
- 3.软链接可以对一个不存在的文件名进行链接
- 4.软链接可以对目录进行链接

**硬链接**：

- 1.硬链接，以文件副本的形式存在。但不占用实际空间。
- 2.不允许给目录创建硬链接
- 3.硬链接只有在同一个文件系统中才能创建

```sh
#创建软链接
ln -s log2013.log link2013
#创建硬链接
ln log2013.log ln2013
```

## 传输文件

linux系统之间传输文件还是非常方便的，主要是一些模型文件很大，可以直接进行传输，不需要下载到本地然后传输。

如果有cpu服务器，也可以做这样传输来做备份。

```sh
#传输文件
scp /home/space/music/1.mp3 root@www.runoob.com:/home/root/others/music   
#传输文件夹
scp -r local_folder remote_username@remote_ip:remote_folder 
```





## 其他常用命令

```sh
pwd          // 查看当前目录路径
ifconfig      //查看网卡信息
ping ip       //查看与某台机器的连接情况
chmod 777     //修改文件权限
----停止防火墙 #停止firewall  #禁止firewall开机启动
systemctl stop firewalld.service 
systemctl disable firewalld.service
```

