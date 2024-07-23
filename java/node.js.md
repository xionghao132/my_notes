# node.js

## node.js概述

> node.js可以在js中接收和处理web请求的应用平台

```
在terminal中输入命令node js文件     //运行文件
```

## node.js模块化编程

```js
concole.los()          //输出
//单独一个文件导出
exports.add=function(a,b){   //编写一个模块
    return a+b;
}
//其他文件引入模块
var demo=require("./demo3");      //引入模块
console.log(demo.add(1,4));       //调用函数
```

## 创建node.js web服务器

```js
//引入node.js的内置模块http 端口8888
var http =require("http");
//创建web服务器并且监听
http.createServer(function(request,response){
    //发送http头部
    //设置响应状态码200
    //设置响应头部信息：Content-Type的类型为纯文本
    response.writeHead(200, {"Content-Type": "text/plain"});

    //发送响应数据
    response.end("Hello World \n");
}).listen(8888);
```

## 处理node.js web请求参数

1.  创建web服务器
2. 引入url模板
3. 利用url解析请求地址的参数和值并且输出
4. 启动测试

```js
//引入node.js的内置模块http
var http =require("http");
var url =require("url");
//创建web服务器并且监听
http.createServer(function(request,response){
    //发送http头部
    //设置响应状态码200
    //设置响应头部信息：Content-Type的类型为纯文本
    response.writeHead(200, {"Content-Type": "text/plain"});
    //解析请求地址
    //参数1：请求地址
    //参数2：true的话使用query到对象中，默认false
    var params=url.parse(request.url,true).query;
    for(var key in params){
        response.write(key+"--"+params[key]);
        response.write("\n");
    }
    //发送响应数据
    response.end("Hello World \n");
}).listen(8888);
console.log("服务器启动成功！");
```

# 包资源管理器npm

## npm的概念

> npm是node包管理和分发工具，可以理解成前端的maven

> 通过npm能很方便下载js库，管理前端工程。而现在node.js已经集成npm工具

```sh
npm -v           //查看当前npm版本
```

> 初始化为npm工程结构

```sh
npm init 
```

> 工程结构可以用来安装js库

- 本地安装：将下载的模块下载到当前目录     `npm install express`

- 全局安装：将下载的模块安装到全局目录     `npm install jquery -g`

  ```sh
  npm root -g         //查看默认下载路径--本地仓库 配置环境变量的时候已经修改
  ```

## npm批量下载

> 类似于后端的maven的pom.xml文件

```sh
snpm install             //根据package.json文件进行更新
```

## 切换镜像源

> 下载nrm

```sh
npm install nrm -g  //管理员打卡命令窗口
nrm ls              //打开列表
nrm use taobao      //更换镜像源
```

![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170234.png)

> 下载cnpm

```sh
npm install -g cnpm --registry=https://registry.npm.tao.org  //安装cnpm
cnpm -v    //查看版本信息
cnpm install 包
```

# webpack

## webpack概念

> 前端资源加载/打包工具，根据模块的依赖关系进行静态分析，然后按指定的规则生成对应的静态资源

> webpack作用：将多个静态资源js,css打包成一个js文件 供以后应用

## webpack下载

```
npm install webpack -g          //一定要用管理员方式运行
npm install webpack-cli -g
webpack -v
```

## webpack打包js文件

实现步骤

1. 创建2两js文件

2. 创建入口文件main.js

   ```js
   var bar=require("./bar");
   var logic=require("./logic");
   bar.info("100+200="+logic.add(100,200));
   ```

3. 创建webpack的配置文件

   ![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170220.png)

   ```js
   var path=require("path");
   
   module.exports={
       entry:"./src/main.js",
       output:{
           //路径
           path:path.resolve(__dirname,"./dist"),
           filename:"bundle.js"
       }
   }
   ```

4. 运行webpack的命令

   ![](https://gitee.com/HB_XN/picture/raw/master/img/20210428170221.png)

5. 创建index.html进行测试

## webpack打包css

 	安装style-loader css-loader组件，创建并且使用css文件，使用webpack命令打包

```sh
cnpm install style-loader css-loader --save-dev
cnpm install less less-loader --save-dev
```

实现步骤

1. 安装转换css的组件         `让require能加载任何形式的文件  实质就是让JavaScript认识css`

2. 修改配置文件

   ```js
   var path=require("path");
   
   module.exports={
       entry:"./src/main.js",
       output:{
           //路径
           path:path.resolve(__dirname,"./dist"),
           filename:"bundle.js"
       },
       //相当于一个转换器进行转换
       module:{
           rules:[
               {
                   test:/\.css$/,
                   use:["style-loader","css-loader"]
               }
           ]
       }
   }
   ```

3. 创建css文件

4. 修改入口文件，加载css文件

   ```js
   var bar=require("./bar");
   var logic=require("./logic");
   require("./css1.css");
   bar.info("100+200="+logic.add(100,200));
   ```

5. 打包并且测试

# es6

## es6概述

 	ecmascript是前端js的语法规范，可以应用在各种环境中，浏览器或者node.js中

