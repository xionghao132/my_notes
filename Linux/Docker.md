[TOC]

# Docker



## 安装Docker

```shell
# 1、yum 包更新到最新 
yum update
# 2、安装需要的软件包， yum-util 提供yum-config-manager功能，另外两个是devicemapper驱动依赖的 
yum install -y yum-utils device-mapper-persistent-data lvm2
# 3、 设置yum源
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
# 4、 安装docker，出现输入的界面都按 y 
yum install -y docker-ce
# 5、 查看docker版本，验证是否验证成功
docker -v

```



## Docker命令

```sh
systemctl start docker       #启动docker
systemctl status docker      #查看docker状态  active为启动状态 dead说明关闭
systemctl stop docker        #关闭docker
systemctl restart docker     #重新启动docker
systemctl enable docker      #开机自动启动
```



## Docker配置镜像

```sh
cd /etc
mkdir docker
cd docker
vi daemon.json

vim /etc/docker/daemon.json
{
    "registry-mirrors": ["https://cq20bk8v.mirror.aliyuncs.com"]
}

systemctl restart docker
```



## Docker镜像命令

```sh
docker images               #查看镜像
docker search redis         #搜索有没有redis镜像，用于下载
```

[docker]: hub.docker.com	"用于查看官方维护的版本"

```sh
docker pusll redis:5.0            #从官方仓库拉取镜像，后面是版本号
docker rmi ID                     #删除镜像，后面是镜像的ID
docker rmi redis:5.0              #当ID一样的时候可以通过名字加版本删除
docker images -q                  #查看所有ID
```



## Doker容器命令

```sh
docker run -it --name=c1 centos:7 /bin/bash #参数i表示一直运行 参数t表示进入容器会有输入命令的终端 name 设置名字
docker run -id --name=c2 centos:7           #d创建容器不会直接进入，exit不会关闭容器
docker exec -it c2 /bin/bash                #进入c2容器，并且分配一个终端
docker stop c2                              #停止c2容器
docker start c2                             #开启容器
docker rm c1                                #删除容器c1
exit                                        #退出容器
docker ps                                   #查看容器
docker ps -a                                #查看历史容器
docker ps -aq                               #显示所有容器的id
docker inspect c2                          #显示容器相关信息 mounts 可以看宿主机和容器数据卷目录

#在主机执行 可以将容器内目录内容复制出来
docker cp id:容器内路径 保存的主机路径
```

## Docker容器的数据卷

> 多个容器挂载同一个目录

```sh
docker run.... -v 宿主机目录：容器内目录（文件）...   #-v参数设置数据卷 宿主机目录必须是绝对路径 可以有多个数据卷
docker run -it --name=c1 -v /root/data:/root/data_container centos:7 #配置一个数据卷
docker run -it --name=c2 -v ~/data2:/root/data2 -v ~/data3:/root/data3 centos:7 #配置多个数据卷
cd ~                                              #进入根目录
```

## Docker容器的数据卷容器

> 多个容器挂载一个容器，这个容器挂载数据卷 同时多个容器也直接挂载在数据卷中

```sh
docker run -it --name=c3 -v /volume centos:7   #设置c3容器
docker run -it --name=c1 --volumes-from c3 centos:7 #将c1容器挂载c3容器中
docker run -it --name=c2 --volumes-from c3 centos:7#将c2容器挂载c3容器中
```

```
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456';
```



## Docker 应用部署

### 部署MySQL

1. 搜索mysql镜像

```shell
docker search mysql
```

2. 拉取mysql镜像

```shell
docker pull mysql:5.6
```

3. 创建容器，设置端口映射、目录映射

```shell
# 在/root目录下创建mysql目录用于存储mysql数据信息
mkdir ~/mysql
cd ~/mysql
```

```shell
docker run -id \
-p 3307:3306 \
--name=c_mysql \
-v $PWD/conf:/etc/mysql/conf.d \
-v $PWD/logs:/logs \
-v $PWD/data:/var/lib/mysql \
-e MYSQL_ROOT_PASSWORD=123456 \
mysql:5.6
```

```sh
docker run -id \
-p 3306:3306 \
--name=c_mysql \
-v $PWD/conf:/etc/mysql/conf.d \
-v $PWD/logs:/logs \
-v $PWD/data:/var/lib/mysql \
-e MYSQL_ROOT_PASSWORD=123456 \
mysql:5.6
```

```sh
docker exec -it c_mysql /bin/bash       #进入mysql容器
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456';
```

第三方

- 参数说明：
  - **-p 3307:3306**：将容器的 3306 端口映射到宿主机的 3307 端口。
  - **-v $PWD/conf:/etc/mysql/conf.d**：将主机当前目录下的 conf/my.cnf 挂载到容器的 /etc/mysql/my.cnf。配置目录
  - **-v $PWD/logs:/logs**：将主机当前目录下的 logs 目录挂载到容器的 /logs。日志目录
  - **-v $PWD/data:/var/lib/mysql** ：将主机当前目录下的data目录挂载到容器的 /var/lib/mysql 。数据目录
  - **-e MYSQL_ROOT_PASSWORD=123456：**初始化 root 用户的密码。

```mysql
mysql> grant all privileges on *.* to 'root'@'%' identified by 'mysql'
    -> ;
Query OK, 0 rows affected, 1 warning (0.00 sec)
mysql> flush privileges;
Query OK, 0 rows affected (0.00 sec)
mysql> 
#8以后的版本
ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY '123456';
flush privileges
```

> 服务器添加防火墙端口3306
>

4. 进入容器，操作mysql

```shell
docker exec –it c_mysql /bin/bash
```

5. 使用外部机器连接容器中的mysql

![](https://gitee.com/HB_XN/picture/raw/master/img/20210501215143.png)



> ==注意==由于mysql8之后的密码策略出现了不同，连接远程得先修改权限。

```python
ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY 'password';
docker restart c_mysql  #重启docker中的mysql，让配置起作用
```

### 部署Tomcat

1. 搜索tomcat镜像

```shell
docker search tomcat
```

2. 拉取tomcat镜像

```shell
docker pull tomcat
```

3. 创建容器，设置端口映射、目录映射

```shell
# 在/root目录下创建tomcat目录用于存储tomcat数据信息
mkdir ~/tomcat
cd ~/tomcat
```

```shell
docker run -id --name=c_tomcat \
-p 8080:8080 \
-v $PWD:/usr/local/tomcat/webapps \
tomcat 
```

- 参数说明：

  - **-p 8080:8080：**将容器的8080端口映射到主机的8080端口

    **-v $PWD:/usr/local/tomcat/webapps：**将主机中当前目录挂载到容器的webapps



4. 使用外部机器访问tomcat

![](https://gitee.com/HB_XN/picture/raw/master/img/20210501215227.png)


### 部署Nginx

1. 搜索nginx镜像

```shell
docker search nginx
```

2. 拉取nginx镜像

```shell
docker pull nginx
```

3. 创建容器，设置端口映射、目录映射


```shell
# 在/root目录下创建nginx目录用于存储nginx数据信息
mkdir ~/nginx
cd ~/nginx
mkdir conf
cd conf
# 在~/nginx/conf/下创建nginx.conf文件,粘贴下面内容
vim nginx.conf
```

```shell
user  nginx;
worker_processes  1;

error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;


events {
    worker_connections  1024;
}


http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile        on;
    #tcp_nopush     on;

    keepalive_timeout  65;

    #gzip  on;

    include /etc/nginx/conf.d/*.conf;
}


```




```shell
docker run -id --name=c_nginx \
-p 80:80 \
-v $PWD/conf/nginx.conf:/etc/nginx/nginx.conf \
-v $PWD/logs:/var/log/nginx \
-v $PWD/html:/usr/share/nginx/html \
nginx
```

- 参数说明：
  - **-p 80:80**：将容器的 80端口映射到宿主机的 80 端口。
  - **-v $PWD/conf/nginx.conf:/etc/nginx/nginx.conf**：将主机当前目录下的 /conf/nginx.conf 挂载到容器的 :/etc/nginx/nginx.conf。配置目录
  - **-v $PWD/logs:/var/log/nginx**：将主机当前目录下的 logs 目录挂载到容器的/var/log/nginx。日志目录

4. 使用外部机器访问nginx

![](https://gitee.com/HB_XN/picture/raw/master/img/20210501215301.png)

### 部署Redis

1. 搜索redis镜像

```shell
docker search redis
```

2. 拉取redis镜像

```shell
docker pull redis:5.0
```

3. 创建容器，设置端口映射

```shell
docker run -id --name=c_redis -p 6379:6379 redis:5.0
```

4. 使用外部机器连接redis

```shell
./redis-cli.exe -h 192.168.215,128 -p 6379      #在window窗口中进入redis目录中执行
```

### 部署elasticsearch

[(56条消息) docker安装elasticsearch（最详细版）_bright的博客-CSDN博客_docker安装elasticsearch](https://blog.csdn.net/qq_40942490/article/details/111594267)

- 查询当前容器：docker container ls -all
- 删除当前容器：docker container rm mycentos(提示: 这一步要确定删除容器没问题的情况下, 才可以做)

### 部署code-server

1. 搜索镜像

```sh
docker search code-server
```

2. 拉取镜像

```sh
docker pull codercom/code-server
```

3. 运行code-server

```sh
# -d 后台运行
# -u 使用root用户来登录容器，这里是避免权限问题
# -p 端口映射
# --name 容器名称
# -v 挂载数据卷 我这里是挂载到home目录下的code，创建这个数据卷的目的是，在本机这里存储编写的代码，防止容器删除了数据丢失
docker run -d -u root -p 8088:8080 --name code-server -v /home/code:/home/code codercom/code-server
```

4. 访问ip:8088

```sh
#因为首次登录code-server是需要密码的，而密码是随机分配的，我们可以进入容器的配置文件去查看
docker exec -it code-server /bin/bash 
#查看密码
cat ~/.config/code-server/config.yaml
#这里会显示密码是什么
```

5. 重启docker容器

```sh
#先退出刚刚登录进去的容器
exit
#重启容器
docker restart code-server
```

重启code-server

[(11条消息) centos用docker安装code-server_zhuzi的博客-CSDN博客](https://blog.csdn.net/weixin_44852935/article/details/113177886)

### 部署anaconda

1. 搜索镜像

```sh
docker search anaconda
```

2. 拉取镜像

```
docker pull continuumio/anaconda3
```

3. 创建容器，配置端口

```sh
docker run -i -t -p 8888:8888 --name anaconda3 -v /home/code:/home/code continuumio/anaconda3 /bin/bash
```

4. 指定目录为虚拟环境

```sh
conda create --prefix ./envs python=3.6
conda activate ./envs
conda deactivate
```

5. 运行jupyter

```sh
jupyter notebook --port 8888 --ip 0.0.0.0 --allow-root
```

```sh
conda install nb_conda_kernels
```

6. 配置镜像源

```sh
conda config --show channels

# 添加清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
 
# 添加阿里云镜像源
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
```

[(11条消息) Docker安装Anaconda环境_Aldnoah-CSDN博客_docker 安装anaconda](https://blog.csdn.net/qq_37119057/article/details/103477395)

### 部署jdk1.8

1. 官网下载jdk1.8  x64   tar.gz

[Java Downloads | Oracle](https://www.oracle.com/java/technologies/downloads/#java8)

2. 解压命令

```sh
tar -zxvf 压缩文件名.tar.gz
```

3. 打开修改/etc/profile配置文件

```sh
vim /etc/profile
```

4. 将文件翻至末尾在最后添加以下代码

```sh
JAVA_HOME=/home/jdk1.8.0_111  #jdk目录
PATH=$JAVA_HOME/bin:$PATH
CLASSPATH=$JAVA_HOME/jre/lib/ext:$JAVA_HOME/lib/tools.jar
export PATH JAVA_HOME CLASSPATH
```

5. 使用source /etc/profile让profile文件立即生效
6. 验证是否安装成功：java -version 和javac -version

### 部署centos

1. 拉取centos:7的镜像

```sh
docker pull centos:7
```

2. 启动centos容器，并把docker上centos的22端口映射到本机50001端口

```sh
docker run -it -p 50001:22 --privileged centos /usr/sbin/init
```

3. 使用容器ID,进入到Centos:

```sh
docker exec -it 容器ID  /bin/bash
```

4. 安装ssh服务和网络必须软件

```sh
 yum install net-tools.x86_64 -y
 yum install -y openssh-server
 yum -y install which.x86_64
 yum -y install vim*          #安装vim
 yum install git -y           #安装git
 yum -y install wget
 yum -y install tmux                    
 #设置代理配置
 git config --global url."https://ghproxy.com/https://github.com".insteadOf "https://github.com"



#目前没有安装成功
yum install autojump
echo '. /usr/share/autojump/autojump.bash' >> ~/.bashrc
cd /etc/profile.d
chmod ugo+x autojump*
source ~/.bashrc
验证安装是否成功
autojump -v
j -v
```

```sh
wegt anaconda
#使用bash命令安装 anaconda
bash ./Anaconda...
#更换python2->python3

vim ~/.bashrc
alias python2="/home/opt/python2.7.5/bin/python2.7"
alias python3="/root/anaconda3/bin/python3.7"
alias python=python3
```

[将CentOS系统python默认版本由python2改为python3_江户川柯壮的博客-CSDN博客](https://blog.csdn.net/edogawachia/article/details/96975093)

5. 安装完后重启SSH服务:

```sh
systemctl restart sshd
```

6. 安装passwd软件，设置centos用户密码

```sh
yum install passwd -y 
```

7.  设置root用户密码

```sh
passwd root
```

8. 注意使用本机IP:50001登录

[Docker中安装Centos_docker_极客研究者-DevPress官方社区 (csdn.net)](https://huaweicloud.csdn.net/63311997d3efff3090b52005.html?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-2-123618150-blog-120636069.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~activity-2-123618150-blog-120636069.pc_relevant_vip_default&utm_relevant_index=3)

### 部署gstore

1. 拉取镜像

```sh
docker pull pkumodlab/gstore-docker:latest #拉取最新版docker镜像

docker pull pkumodlab/gstore:0.9
```

2. 启动镜像

```sh
docker run -itd -p 9000:29000 9ca4388fc81e /bin/bash 
# 将9000端口映射到宿主机的29000端口
# 9ca4388fc81e 为image id
```

3. 进入容器

```sh
docker exec -it a5016bd46094 /bin/bash #推荐采用exec方式进入容器，退出后容器不会停止
```

4. 启动api服务

```sh
bin/ghttp [db_name] [port]
nohup bin/ghttp [db_name] [port] & #后台启动

#bin/ghttp -db lubm -p 9000 -c enable
```

5. 实例化数据库

```sh
bin/gbuild -db lubm -f ./data/lubm/lubm.nt
```

6. 开启ghttp服务

```
nohup ./bin/ghttp  9000 &  #默认就是9000端口
```

7. 关闭ghttp服务

```sh
bin/shutdown
```

8. 开启gserver服务

```sh
bin/gserver -s
```

9. 关闭gserver服务

```
bin/gserver -t
```

[#gStore-weekly | Centos7系统下gStore在 docker上的安装部署 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/395833759)

[图数据库引擎gStore系统](https://www.gstore.cn/pcsite/index.html#/documentation)

[New Issue · pkumod/gStore (github.com)](https://github.com/pkumod/gStore/issues/new)

## Dockerfile

### 镜像制作

```sh
docker commit ID 镜像名称：版本号        	#容器转化为镜像文件  数据卷目录挂载不能写到镜像中
docker save -o 压缩文件名称 镜像名称：版本号  #将镜像变成压缩文件
docker load -i 压缩文件名称                #将压缩文件转化为镜像
```

> dockerfile是一个文本文件

| 关键字      | 作用                     | 备注                                                         |
| ----------- | ------------------------ | ------------------------------------------------------------ |
| FROM        | 指定父镜像               | 指定dockerfile基于那个image构建                              |
| MAINTAINER  | 作者信息                 | 用来标明这个dockerfile谁写的                                 |
| LABEL       | 标签                     | 用来标明dockerfile的标签 可以使用Label代替Maintainer 最终都是在docker image基本信息中可以查看 |
| RUN         | 执行命令                 | 执行一段命令 默认是/bin/sh 格式: RUN command 或者 RUN ["command" , "param1","param2"] |
| CMD         | 容器启动命令             | 提供启动容器时候的默认命令 和ENTRYPOINT配合使用.格式 CMD command param1 param2 或者 CMD ["command" , "param1","param2"] |
| ENTRYPOINT  | 入口                     | 一般在制作一些执行就关闭的容器中会使用                       |
| COPY        | 复制文件                 | build的时候复制文件到image中                                 |
| ADD         | 添加文件                 | build的时候添加文件到image中 不仅仅局限于当前build上下文 可以来源于远程服务 |
| ENV         | 环境变量                 | 指定build时候的环境变量 可以在启动的容器的时候 通过-e覆盖 格式ENV name=value |
| ARG         | 构建参数                 | 构建参数 只在构建的时候使用的参数 如果有ENV 那么ENV的相同名字的值始终覆盖arg的参数 |
| VOLUME      | 定义外部可以挂载的数据卷 | 指定build的image那些目录可以启动的时候挂载到文件系统中 启动容器的时候使用 -v 绑定 格式 VOLUME ["目录"] |
| EXPOSE      | 暴露端口                 | 定义容器运行的时候监听的端口 启动容器的使用-p来绑定暴露端口 格式: EXPOSE 8080 或者 EXPOSE 8080/udp |
| WORKDIR     | 工作目录                 | 指定容器内部的工作目录 如果没有创建则自动创建 如果指定/ 使用的是绝对地址 如果不是/开头那么是在上一条workdir的路径的相对路径 |
| USER        | 指定执行用户             | 指定build或者启动的时候 用户 在RUN CMD ENTRYPONT执行的时候的用户 |
| HEALTHCHECK | 健康检查                 | 指定监测当前容器的健康监测的命令 基本上没用 因为很多时候 应用本身有健康监测机制 |
| ONBUILD     | 触发器                   | 当存在ONBUILD关键字的镜像作为基础镜像的时候 当执行FROM完成之后 会执行 ONBUILD的命令 但是不影响当前镜像 用处也不怎么大 |
| STOPSIGNAL  | 发送信号量到宿主机       | 该STOPSIGNAL指令设置将发送到容器的系统调用信号以退出。       |
| SHELL       | 指定执行脚本的shell      | 指定RUN CMD ENTRYPOINT 执行命令的时候 使用的shell            |

> vim   centos_dockerfile         dockerfile创建完成后使用如下命令

```sh
docker build -f ./centos_dockerfile -t xiong_centos:1 .     #f表明指定路径 t表明新的镜像的名称和版本 .表明是路径
```

> 部署springboot项目    springboot_dockerfile

```sh
FROM java:8
MAINTAINER xionghao <1329424972@qq.com>
ADD springboot-hello-0.0.1-SNAPSHOT.jar app.jar
CMD java -jar  app.jar
```

```sh
docker run -id -p 9000:8080 app
```



## Docker Compose

### 安装Docker Compose

```shell
# Compose目前已经完全支持Linux、Mac OS和Windows，在我们安装Compose之前，需要先安装Docker。下面我 们以编译好的二进制包方式安装在Linux系统中。 
curl -L https://github.com/docker/compose/releases/download/1.22.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose
# 设置文件可执行权限 
chmod +x /usr/local/bin/docker-compose
# 查看版本信息 
docker-compose -version
```

### 卸载Docker Compose

```shell
# 二进制包方式安装的，删除二进制文件即可
rm /usr/local/bin/docker-compose
```

###  使用docker compose编排nginx+springboot项目

1. 创建docker-compose目录

```shell
mkdir ~/docker-compose
cd ~/docker-compose
```

2. 编写 docker-compose.yml 文件

```shell
version: '3'
services:
  nginx:
   image: nginx
   ports:
    - 80:80
   links:
    - app
   volumes:
    - ./nginx/conf.d:/etc/nginx/conf.d
  app:
    image: app
    expose:
      - "8080"
```

3. 创建./nginx/conf.d目录

```shell
mkdir -p ./nginx/conf.d
```



4. 在./nginx/conf.d目录下 编写itheima.conf文件

```shell
server {
    listen 80;
    access_log off;

    location / {
        proxy_pass http://app:8080;
    }
   
}
```

5. 在~/docker-compose 目录下 使用docker-compose 启动容器

```shell
docker-compose up
```

6. 测试访问

```shell
http://192.168.149.135/hello
```

## Windows Docker Desktop

打开desktop报错，连接不上

```sh
#管理员权限cmd
netsh winsock reset

Net stop com.docker.service  #重启docker服务
Net start com.docker.service
#powershell 查看
wsl -l -v 
```

```sh
# 使用curl命令报错，raw.githubusercontent.com无法访问的问题  PlugInstall 安装失败
# 再加入一个github.com 的hosts文件中
git config --global --unset http.proxy 
git config --global --unset https.proxy
```

[Githubusercontent - raw.Githubusercontent.com (ipaddress.com)](https://www.ipaddress.com/site/raw.githubusercontent.com)
