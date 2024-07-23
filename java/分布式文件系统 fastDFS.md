# 分布式文件系统 fastDFS

## 概念

 	FastDFS是用c语言编写的一款开源的分布式文件系统，它是由淘宝资深架构师余庆编写并开源。FastDFS专为互联 

网量身定制，充分考虑了冗余备份、负载均衡、线性扩容等机制，并注重高可用、高性能等指标，使用FastDFS很 

容易搭建一套高性能的文件服务器集群提供文件上传、下载等服务。 

**为什么要使用fastDFS呢？** 

 	NFS、GFS都是通用的分布式文件系统，通用的分布式文件系统的优点的是开发体验好，但是系统复杂 

性高、性能一般，而专用的分布式文件系统虽然开发体验性差，但是系统复杂性低并且性能高。fastDFS非常适合 

存储图片等那些小文件，fastDFS不对文件进行分块，所以它就没有分块合并的开销，fastDFS网络通信采用 

socket，通信速度很快。

## 工作原理

![image-20220106175613738](D:\my_ty_file\images\image-20220106175613738.png)

1. **Tracker** 

Tracker Server作用是负载均衡和调度，通过Tracker server在文件上传时可以根据一些策略找到Storage server提 

供文件上传服务。可以将tracker称为追踪服务器或调度服务器。 

FastDFS集群中的Tracker server可以有多台，Tracker server之间是相互平等关系同时提供服务，Tracker server 

不存在单点故障。客户端请求Tracker server采用轮询方式，如果请求的tracker无法提供服务则换另一个tracker。 

2. **Storage**

Storage Server作用是文件存储，客户端上传的文件最终存储在Storage服务器上，Storage server没有实现自己的 

文件系统而是使用操作系统的文件系统来管理文件。可以将storage称为存储服务器。Storage集群采用了分组存储方式。storage集群由一个或多个组构成，集群存储总容量为集群中所有组的存储容量 

之和。一个组由一台或多台存储服务器组成，组内的Storage server之间是平等关系，不同组的Storage server之 

间不会相互通信，同组内的Storage server之间会相互连接进行文件同步，从而保证同组内每个storage上的文件完 

全一致的。一个组的存储容量为该组内存储服务器容量最小的那个，由此可见组内存储服务器的软硬件配置最好是 

一致的。 

采用分组存储方式的好处是灵活、可控性较强。比如上传文件时，可以由客户端直接指定上传到的组也可以由 

tracker进行调度选择。一个分组的存储服务器访问压力较大时，可以在该组增加存储服务器来扩充服务能力（纵向 

扩容）。当系统容量不足时，可以增加组来扩充存储容量（横向扩容）。 

3. **Storage状态收集**

Storage server会连接集群中所有的Tracker server，定时向他们报告自己的状态，包括磁盘剩余空间、文件同步 

状况、文件上传下载次数等统计信息。



## 文件上传结果

> group1/M00/00/00/rBEAA2HV41qAW4moAABChuAgOPs2051721.jpg

* 组名：文件上传后所在的storage组名称，在文件上传成功后有storage服务器返回，需要客户端自行保存。虚拟磁盘路径：storage配置的虚拟路径，与磁盘选项store_path*对应。如果配置了store_path0则是M00， 

如果配置了store_path1则是M01，以此类推。 

* 数据两级目录：storage服务器在每个虚拟磁盘路径下创建的两级目录，用于存储数据文件。 

* 文件名：与文件上传时不同。是由存储服务器根据特定信息生成，文件名包含：源存储服务器IP地址、文件创 

建时间戳、文件大小、随机数和文件拓展名等信息。 

之后可以通过 http://ip:8888/fileid 就可以访问到资源。

##  fastDFS安装

* 使用docker进行部署安装

```sh
docker pull delron/fastdfs
```

* 部署tracer,默认端口是22122

```sh
docker run -d --name tracker -p 22122:22122 -v /Users/zzs/develop/temp/tracker:/var/fdfs delron/fastdfs tracker 
```

* 部署storage, 前面一个端口绑定的是nginx的访问端口，默认是8888，后面一个端口绑定的是storage的端口，默认是23000

```sh
docker run -d --name storage -p 8888:8888 -p 23000:23000 -e TRACKER_SERVER=123.57.254.35:22122 -v /Users/zzs/develop/temp/storage:/var/fdfs -e GROUP_NAME=group1 delron/fastdfs storage
```

* 如果要修改storage的端口，还要进入容器修改配置文件。

```sh
docker exec -it 2bc9f8268eda bash
```

* 进入storeage的容器，在==/etc/fdfs== 下有storage.conf中的最后一行有以下内容，可以进行修改

```sh
# the port of the web server on this storage server
http.server_port=8888
```

* 还要修改nginx的配置文件，在==/usr/local/nginx/conf/==下。

## Springboot整合fastDFS

1. maven依赖

==注意：==fastDFS依赖    官网https://github.com/happyfish100/FastDFS

* 解压缩后进入项目，cmd 。
*  使用命令 maven clean install 打包文件 然后进入target目录中找到包。
* 进入自己的项目，打开项目结构，添加旁边有个加入jar。

```xml
 <dependencies>
        <!--        springMVC-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <version>2.3.6.RELEASE</version>
        </dependency>
        <!--        springboot test-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <version>2.3.11.RELEASE</version>
            <scope>test</scope>
        </dependency>
<!--        lombok-->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <version>1.18.14</version>
        </dependency>
        <!--        fastdfs 注意这个是一个jar包 要先添加-->
        <dependency>
            <groupId>org.csource</groupId>
            <artifactId>fastdfs-client-java</artifactId>
            <version>1.29-SNAPSHOT</version>
        </dependency>

        <!-- https://mvnrepository.com/artifact/commons-io/commons-io -->
        <dependency>
            <groupId>commons-io</groupId>
            <artifactId>commons-io</artifactId>
            <version>2.4</version>
        </dependency>
<!--        mybatis3-->
        <dependency>
            <groupId>com.baomidou</groupId>
            <artifactId>mybatis-plus-boot-starter</artifactId>
            <version>3.4.0</version>
        </dependency>
<!--        mysql数据库连接-->
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version> 8.0.15</version>
        </dependency>
    </dependencies>
```

2. fastDFS配置文件(从官网的test中可以找到这个文件)

```properties
connect_timeout = 2
network_timeout = 30 
charset = UTF-8
http.tracker_http_port = 8888  #这个得注意和自己写的一样
http.anti_steal_token = no
http.secret_key = FastDFS1234567890

#可以写多个ip 看官网
tracker_server = 123.57.254.35:22122
```

3. FastDFSClientUtils工具类

```java
public class FastDFSClientUtils {
    private TrackerClient trackerClient = null;
    private TrackerServer trackerServer = null;
    private StorageServer storageServer = null;
    private StorageClient1 storageClient = null;
    // 初始化服务器和客户端
    public FastDFSClientUtils(String conf) throws Exception {
        if (conf.contains("classpath:")) {
            conf = conf.replace("classpath:", this.getClass().getResource("/").getPath());
        }
        //加载fastDFS客户端
        ClientGlobal.init(conf);
        //创建tracker客户端
         trackerClient = new TrackerClient();
          trackerServer = trackerClient.getTrackerServer();
        storageServer = null;
        //定义storage客户端
        storageClient = new StorageClient1(trackerServer, storageServer);
    }
    /**
     *@Description: 通过本地文件目录上传文件
     *@Param: [java.lang.String, java.lang.String, org.csource.common.NameValuePair[]]
     *@return: java.lang.String
     *@Author: 熊浩
     *@date: 2022/1/6
     */
    public String uploadFile(String fileName, String extName, NameValuePair[] metas) {
        String result=null;
        try {
            result = storageClient.upload_file1(fileName, extName, metas);
        }catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }
    /**
     *@Description: 通过文件字节方式上传文件
     *@Param: [byte[], java.lang.String, org.csource.common.NameValuePair[]]
     *@return: java.lang.String
     *@Author: 熊浩
     *@date: 2022/1/6
     */
    public String uploadFile(byte[] fileContent, String extName, NameValuePair[] metas) {
        String result=null;
        try {
            result = storageClient.upload_file1(fileContent, extName, metas);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;
    }
    /**
     *@Description: 下载文件
     *@Param: [java.lang.String]
     *@return: byte[]
     *@Author: 熊浩
     *@date: 2022/1/6
     */
    public byte[] download_bytes(String path) {
        byte[] b=null;
        try {
            b = storageClient.download_file1(path);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return b;
    }
    /**
     *@Description: 通过fileid删除fastDFS中的文件
     *@Param: [java.lang.String]
     *@return: java.lang.Integer
     *@Author: 熊浩
     *@date: 2022/1/6
     */
    public Integer delete_file(String storagePath){
        int result=2;
        try {
            result = storageClient.delete_file1(storagePath);  //0 是删除成功 2是删除失败
        } catch (Exception e) {
            e.printStackTrace();
        }
        return  result;
    }
    public FileInfo query_info(String storagePath){
        FileInfo fileInfo = null;
        try {
            fileInfo = storageClient.query_file_info1(storagePath);
            System.out.println(fileInfo);
            // 查询文件元信息
//            NameValuePair[] metadata1 = storageClient.get_metadata1("group1/M00/00/00/rBEAA2HVM-CAGa7UAACOMSywWxw987.jpg");
//            System.out.println(metadata1);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return fileInfo;
    }
}
```

