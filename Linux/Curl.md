# Curl

## 概述

curl 命令是一个在 Linux 系统中利用 URL 工作的命令行文件传输工具，常用于服务访问和文件下载。curl 支持 HTTP、HTTPS、FTP 等多种协议（默认是 HTTP 协议），可用于模拟服务请求以及上传和下载文件。

## 使用

### 快速使用

- curl 命令的标准语法： `curl [options] [url]`
- 不使用参数项执行请求： `curl http://www.baidu.com`

### 参数

常见参数项包括：

- -i ：显示响应头信息
- -o ：将请求结果写入到指定文件中
- -s ：静默模式，不显示额外信息
- -w ：指定输出内容格式



如测试接口是否正常： `curl -o /dev/null -s -w %{http_code} http://www.baidu.com`

## 模拟GET/POST请求

### GET请求

- 直接使用 curl 无参请求方式默认为 GET 请求，如：` curl http://localhost:8080/getUserInfo?id=1`
- 还可以使用 -X 参时来指定请求方式为 GET



### POST请求

POST 请求时，可以使用以下参数：

- -X ：指定请求方式（如 POST）
- -H ：指定请求头信息（如 “Content-Type:application/json”）,**-H 指定 headers 头的时候需要单个使用，即一个 -H 指定一个头字段信息**
- -d ：指定请求参数内容（可以使用多次，或一次指定多个参数，甚至传递 json 对象；还可以使用文件作为参数）

```sh
### 指定地址、请求头信息、请求类型、请求参数
curl 'http://localhost:8080/cnd_inke/qc/v2/inke' \
-H "Content-Type:application/json" \
-H 'Authorization:bearer' \
-X POST \
-d '{"Id":"12330245","visitTimes":1,"docType":"散文","docId":"36e5854f5f0e4f80b7ccc6c52c063243"}'


-d '@/test.json'   #以文件作为参数也可以
```



## 上传和下载文件

### 文件上传

使用 `-F` 参数上传文件： `curl -F 'file=@test.png;type=image/png' http://www.baidu.com/upload`

- 使用 `-F` 参数时，默认使用文件上传格式
- 可指定多个文件和文件类型，用 `;` 分隔
- 

### 文件下载

Curl 下载文件时使用 -O 选项，默认使用网络文件的名字作为本地文件名。
文件下载方式有：

可以使用重定向保存到指定文件： … >> index.html
如果想要为下载的文件指定名称，则使用 -o 代替 -O，curl -o file2.pdf www.example.com/file.pdf
-# ，显示下载进度和速度等信息，可使用 -s 关闭显示
--limit-rate ，设置下载时最大下载速度，如 --limit-rate 1m
使用 -C 参数可以设置开启断点续传

```sh
curl http://mirrors.163.com/centos/8.1.1911/isos/x86_64/CentOS-8.1.1911-x86_64-dvd1.iso
```

批量下载文件

curl 还支持下载多个文件，只需要多次指定 -O 和文件地址即可，如 curl -O [URL1] -O [URL2] -O [URL3] ...
如果多个文件地址符合规律，可以使用正则来批量下载：`curl -O ftp://ftp.example.com/file[1-30].jpg`

### 使用ip代理请求服务

#### 设置代理信息

curl 命令还可以使用 `-x` 参数来设置 http(s) 代理、socks 代理，设置用户名、密码、认证信息方式如下

```
# 使用HTTP代理访问；如果未指定端口，默认使用8080端口;
# protocol 默认为 http_proxy，其他可能的值包括：
# http_proxy、HTTPS_PROXY、socks4、socks4a、socks5；
# 如： --proxy 8.8.8.8:8080； -x "http_proxy://aiezu:123@aiezu.com:80"
-x host:port
-x [protocol://[user:pwd@]host[:port]
--proxy [protocol://[user:pwd@]host[:port]
```



- 参数 -x 与 --proxy 等价
- 如果未指定端口，默认使用 8080 端口
- protocol 协议默认为 http_proxy 代理



#### 使用 ip 代理示例

```sh
# 指定 http 代理 IP 和端口
curl -x 113.185.19.192:80 http://baidu.com
#显式指定为 http 代理
curl -x http_proxy://113.185.19.192:80 http://baidu.com
 
#指定 https 代理
curl -x HTTPS_PROXY://113.185.19.192:80 http://baidu.com
 
#指定代理用户名和密码，basic 认证方式
curl -x aiezu:123456@113.185.19.192:80 http://baidu.com
curl -x 113.185.19.192:80 -U aiezu:123456 http://baidu.com
#指定代理协议、用户名和密码，basic 认证方式
curl -x HTTPS_PROXY://aiezu:123456@113.185.19.192:80 http://baidu.com
 
#指定代理用户名和密码，ntlm 认证方式
curl -x 113.185.19.192:80 -U aiezu:123456 --proxy-ntlm http://baidu.com

```

[Linux 从入门到精通：curl 命令使用详解_linux curl用法-CSDN博客](https://blog.csdn.net/grammer_du/article/details/130451087)

## PyCurl

### GET/POST请求

```python
import pycurl

# 创建一个Curl对象
c = pycurl.Curl()

# 设置请求的URL
c.setopt(c.URL, 'https://www.example.com')

# 设置请求方法为GET
c.setopt(c.HTTPGET, 1)
# 设置请求方法为POST
#c.setopt(c.POST, 1)

# 设置回调函数，以便获取响应数据
data = []
def collect_data(chunk):
    data.append(chunk)

c.setopt(c.WRITEFUNCTION, collect_data)

# 发送请求
c.perform()

# 获取响应数据
response = b''.join(data)

# 打印响应数据
print(response.decode('utf-8'))
```



### 文件上传

```python
import pycurl

# 创建一个Curl对象
c = pycurl.Curl()

# 设置请求的URL
c.setopt(c.URL, 'https://www.example.com/api/upload_file')

# 设置请求方法为POST
c.setopt(c.POST, 1)

# 设置请求数据，并包括文件上传
post_data = [    ('username', 'example_user'),    ('file', (c.FORM_FILE, 'path/to/file.txt'))]
c.setopt(c.HTTPPOST, post_data)

# 设置回调函数，以便获取响应数据
data = []
def collect_data(chunk):
    data.append(chunk)

c.setopt(c.WRITEFUNCTION, collect_data)

# 发送请求
c.perform()

# 获取响应数据
response = b''.join(data)

# 打印响应数据
print(response.decode('utf-8'))
```

