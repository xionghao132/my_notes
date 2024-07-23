[TOC]

# Crawler

## 爬虫入门程序

### 环境准备

> pom.xml

```java
<dependencies>
    <!-- HttpClient -->
    <dependency>
        <groupId>org.apache.httpcomponents</groupId>
        <artifactId>httpclient</artifactId>
        <version>4.5.3</version>
    </dependency>

    <!-- 日志 -->
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-log4j12</artifactId>
        <version>1.7.25</version>
    </dependency>
</dependencies>
```

> log4j.properties

```java
log4j.rootLogger=DEBUG,A1
log4j.logger.cn.itcast = DEBUG

log4j.appender.A1=org.apache.log4j.ConsoleAppender
log4j.appender.A1.layout=org.apache.log4j.PatternLayout
log4j.appender.A1.layout.ConversionPattern=%-d{yyyy-MM-dd HH:mm:ss,SSS} [%t] [%c]-[%p] %m%n
```

### 代码编写

```java
 public static void main(String[] args) throws IOException {
        //创建HttpClient对象
        CloseableHttpClient httpClient = HttpClients.createDefault();
        //创建HttpGet请求
        HttpGet httpGet = new HttpGet("http://www.itcast.cn/");
        CloseableHttpResponse response = null;
        try {
            //使用HttpClient发起请求
            response = httpClient.execute(httpGet);
            //判断响应状态码是否为200
            if (response.getStatusLine().getStatusCode() == 200) {
                //如果为200表示请求成功，获取返回数据
                String content = EntityUtils.toString(response.getEntity(), "UTF-8");
                //打印数据长度
                System.out.println(content);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            //释放连接
            if (response == null) {
                try {
                    response.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                httpClient.close();
            }
        }
    }
```

### HttpClient

#### Get请求

> Get请求入门程序为带参数的    以下为不带参数的

```java
//创建HttpGet请求
URIBuilder uriBuilder=new URIBuilder("http://yun.itheima.com/search");
//设置多组参数
uriBuilder.setParameter("keys", "java").setParameter("keys", "python"); 
HttpGet httpGet = new HttpGet(uriBuilder.build());
```

#### Post请求

> 不带参数的post的请求

```java
HttpPost httpPost=new HttpPost("http://yun.itheima.com/search");
```

> 带参数的post的请求

```java
HttpPost httpPost = new HttpPost("http://www.itcast.cn/");
List<NameValuePair>params=new ArrayList<>();
params.add(new BasicNameValuePair("keys","java"));
//第一个参数是封装好的表单数据
UrlEncodedFormEntity formEntity=new UrlEncodedFormEntity(params,"utf8");
httpPost.setEntity(formEntity);
```

#### 连接池

> doGet是自己实现的方法，与上面的主方法类似

```java
//创建连接池管理器
PoolingHttpClientConnectionManager cm=new PoolingHttpClientConnectionManager();
//设置最大连接数
cm.setMaxTotal(100);
//设置最大连接主机数
cm.setDefaultMaxPerRoute(10);
doGet(cm);
//使用连接池管理器发起请求
```

==注意：==`doGet`方法不能关闭httpclient

```java
//不能关闭HttpClient
//httpClient.close();
```

#### 请求参数

```java
 RequestConfig requestConfig = RequestConfig.custom()
            .setConnectTimeout(1000)//设置创建连接的最长时间
            .setConnectionRequestTimeout(500)//设置获取连接的最长时间
            .setSocketTimeout(10 * 1000)//设置数据传输的最长时间
            .build();
httpGet.setConfig(requestConfig);
```

## Jsoup

### 环境准备

```java
		<dependency>
            <groupId>org.jsoup</groupId>
            <artifactId>jsoup</artifactId>
            <version>1.13.1</version>
        </dependency>

        <dependency>
            <groupId>commons-io</groupId>
            <artifactId>commons-io</artifactId>
            <version>2.6</version>
        </dependency>

        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-lang3</artifactId>
            <version>3.7</version>
 		</dependency>
```

### Jsoup解析url

```java
	@Test
    public void test() throws Exception{
        //解析url地址
        Document document = Jsoup.parse(new URL("http://www.itcast.cn/"), 1000);
        //获取title的内容
        Element title = document.getElementsByTag("title").first();
        System.out.println(title.text());
    }
```

### Jsoup解析字符串

```java
String content = FileUtils.readFileToString(new File("C:\\Users\\我想静静\\Desktop\\test.html"), "utf8");
//解析字符串
Document doc = Jsoup.parse(content);
```

### Jsoup解析文件

```java
Documnet doc=Jsoup.parse(new File("C:\\Users\\我想静静\\Desktop\\test.html"), "utf8");
```

### 获取元素

1.  根据id查询元素getElementById
2. 根据标签获取元素getElementsByTag
3. 根据class获取元素getElementsByClass
4. 根据属性获取元素getElementsByAttribute

> Id是直接获取到一个元素 而其他的三个获取到的是元素数组

```java
//1.    根据id查询元素getElementById
Element element = document.getElementById("city_bj");
//2.   根据标签获取元素getElementsByTag
element = document.getElementsByTag("title").first();
//3.   根据class获取元素getElementsByClass
element = document.getElementsByClass("s_name").last();
//4.   根据属性获取元素getElementsByAttribute
element = document.getElementsByAttribute("abc").first();
element = document.getElementsByAttributeValue("class", "city_con").first();
```

### 元素中获取数据

- 从元素中获取id
-  从元素中获取className
-  从元素中获取属性的值attr
-  从元素中获取所有属性attributes
-  从元素中获取文本内容text

```java
//获取元素
Element element = document.getElementById("test");
//1.   从元素中获取id
String str = element.id();
//2.   从元素中获取className
str = element.className();
//3.   从元素中获取属性的值attr
str = element.attr("id");
//4.   从元素中获取所有属性attributes
str = element.attributes().toString();
//5.   从元素中获取文本内容text
str = element.text();
```

### Selector选择器概述

1. ***\*tagname\****: 通过标签查找元素，比如：span
2. ***\*#id\****: 通过ID查找元素，比如：# city_bj
3. ***\*.class\****: 通过class名称查找元素，比如：.class_a
4. ***\*[attribute]\****: 利用属性查找元素，比如：[abc]
5. ***\*[attr=value]\****: 利用属性值来查找元素，比如：[class=s_name]

```java
//tagname: 通过标签查找元素，比如：span
Elements span = document.select("span");
for (Element element : span) {
    System.out.println(element.text());
}
//#id: 通过ID查找元素，比如：#city_bjj
String str = document.select("#city_bj").text();
//.class: 通过class名称查找元素，比如：.class_a
str = document.select(".class_a").text();
//[attribute]: 利用属性查找元素，比如：[abc]
str = document.select("[abc]").text();
//[attr=value]: 利用属性值来查找元素，比如：[class=s_name]
str = document.select("[class=s_name]").text();
```

### Selector选择器组合使用

1. ***\*el#id\****: 元素+ID，比如： h3#city_bj

2. ***\*el.class\****: 元素+class，比如： li.class_a

3. ***\*el[attr]\****: 元素+属性名，比如： span[abc]

4. ***\*任意组合\****: 比如：span[abc].s_name

5. ***\*ancestor child\****: 查找某个元素下子元素，比如：.city_con li 查找"city_con"下的所有li

6. ***\*parent > child\****: 查找某个父元素下的直接子元素，比如：

7. .city_con > ul > li 查找city_con第一级（直接子元素）的ul，再找所有ul下的第一级li

8. ***\*parent > \*\****: 查找某个父元素下所有直接子元素

   ==注意：直接子元素是最近的一层==

```java
//el#id: 元素+ID，比如： h3#city_bj
String str = document.select("h3#city_bj").text();
//el.class: 元素+class，比如： li.class_a
str = document.select("li.class_a").text();
//el[attr]: 元素+属性名，比如： span[abc]
str = document.select("span[abc]").text();
//任意组合，比如：span[abc].s_name
str = document.select("span[abc].s_name").text();
//ancestor child: 查找某个元素下子元素，比如：.city_con li 查找"city_con"下的所有li
str = document.select(".city_con li").text();
//parent > child: 查找某个父元素下的直接子元素，
//比如：.city_con > ul > li 查找city_con第一级（直接子元素）的ul，再找所有ul下的第一级li
str = document.select(".city_con > ul > li").text();
//parent > * 查找某个父元素下所有直接子元素.city_con > *
str = document.select(".city_con > *").text();
```

### 爬虫案例

## WebMagic

### WebMagic四大组件

1. **Downloader**

    	Downloader负责从互联网上下载页面，以便后续处理。WebMagic默认使用了Apache HttpClient作为下载工具。 

2. **PageProcessor**

    	PageProcessor负责解析页面，抽取有用信息，以及发现新的链接。WebMagic使用Jsoup作为HTML解析工具，并基于其开发了解析XPath的工具Xsoup。在这四个组件中，PageProcessor对于每个站点每个页面都不一样，是需要使用者定制的部分。

3. **Scheduler**

    	Scheduler负责管理待抓取的URL，以及一些去重的工作。WebMagic默认提供了JDK的内存队列来管理URL，并用集合来进行去重。也支持使用Redis进行分布式管理。

4. **Pipeline**

    	Pipeline负责抽取结果的处理，包括计算、持久化到文件、数据库等。WebMagic默认提供了“输出到控制台”和“保存到文件”两种结果处理方案。Pipeline定义了结果保存的方式，如果你要保存到指定数据库，则需要编写对应的Pipeline。对于一类需求一般只需编写一个Pipeline。

### 数据流转的对象

1. **Request**

     Request是对URL地址的一层封装，一个Request对应一个URL地址。它是PageProcessor与Downloader交互的载体，也是PageProcessor控制Downloader唯一方式。除了URL本身外，它还包含一个Key-Value结构的字段extra。你可以在extra中保存一些特殊的属性，然后在其他地方读取，以完成不同的功能。例如附加上一个页面的一些信息等。

2. **Page**

    Page代表了从Downloader下载到的一个页面——可能是HTML，也可能是JSON或者其他文本格式的内容。 Page是WebMagic抽取过程的核心对象，它提供一些方法可供抽取、结果保存等。

3. **ResultItems**

    ResultItems相当于一个Map，它保存PageProcessor处理的结果，供Pipeline使用。它的API与Map很类似，值得注意的是它有一个字段skip，若设置为true，则不应被Pipeline处理。

### 入门程序

```java
public class JobProcessor implements PageProcessor {
    //解析页面
    public void process(Page page) {
        //解析返回的数据page,并且把解析的结果放到ResultItems中
      page.putField("div",page.getHtml().css("a.link-login").all());
    }
    private Site site=Site.me();
    public Site getSite() {
        return site;
    }
    public static void main(String[] args) {
        //设置要爬取的页面并执行
        Spider.create(new JobProcessor()).addUrl("https://www.jd.com/").run();
    }
}
```

### 元素选择Selectable

 	WebMagic里主要使用了三种抽取技术：XPath、正则表达式和CSS选择器。另外，对于JSON格式的内容，可使用JsonPath进行解析。

1. **XPath**

    以上是获取属性class=mt的div标签，里面的h1标签的内容

```
page.getHtml().xpath("//div[@class=mt]/h1/text()")
```

 	也可以参考W3School离线手册(2017.03.11版).chm

2. **CSS选择器**

    CSS选择器是与XPath类似的语言。在上一次的课程中，我们已经学习过了Jsoup的选择器，它比XPath写起来要简单一些，但是如果写复杂一点的抽取规则，就相对要麻烦一点。

    div.mt>h1表示class为mt的div标签下的直接子元素h1标签

```java
page.getHtml().css("div.mt>h1").toString() 
```

 	可是使用:nth-child(n)选择第几个元素，如下选择第一个元素

```java
page.getHtml().css("div#news_div > ul > li:nth-child(1) a").toString()
```

==注意：==需要使用>，就是直接子元素才可以选择第几个元素

3. **正则表达式**

    正则表达式则是一种通用的文本抽取语言。在这里一般用于获取url地址。

### 获取结果API

|    方法    |               说明                |                 示例                 |
| :--------: | :-------------------------------: | :----------------------------------: |
|   get()    |     返回一条String类型的结果      |   String link= html.links().get()    |
| toString() | 同get()，返回一条String类型的结果 | String link= html.links().toString() |
|   all()    |         返回所有抽取结果          |    List links= html.links().all()    |

> 当有多条数据的时候，使用get()和toString()都是获取第一个url地址。

### 获取链接

```java
//links()能获取当前标签下所有链接  regex()正则表达式对链接进行筛选 addTargetRequests添加链接到Schedulercheduler
page.addTargetRequests(page.getHtml().css("ul#navitems-group1").links().regex(".*miaosha.*").all());
page.putField("url",page.getHtml().css("li#ttbar-home"));
```

### 用Pipeline保存结果

```java
Spider.create(new JobProcessor())
            //初始访问url地址
            .addUrl("https://www.jd.com/moreSubject.aspx")
            .addPipeline(new FilePipeline("D:/webmagic/"))    //使用文件夹输出
            .thread(5)//设置线程数
            .run();
```

### 爬虫配置、启动、终止

> Spider是爬虫启动的入口。在启动爬虫之前，我们需要使用一个PageProcessor创建一个Spider对象，然后使用run()进行启动。

> 同时Spider的其他组件（Downloader、Scheduler、Pipeline）都可以通过set方法来进行设置。

| ***\*方法\****            | ***\*说明\****                                   | ***\*示例\****                                               |
| ------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| create(PageProcessor)     | 创建Spider                                       | Spider.create(new GithubRepoProcessor())                     |
| addUrl(String…)           | 添加初始的URL                                    | spider .addUrl("http://webmagic.io/docs/")                   |
| thread(n)                 | 开启n个线程                                      | spider.thread(5)                                             |
| run()                     | 启动，会阻塞当前线程执行                         | spider.run()                                                 |
| start()/runAsync()        | 异步启动，当前线程继续执行                       | spider.start()                                               |
| stop()                    | 停止爬虫                                         | spider.stop()                                                |
| addPipeline(Pipeline)     | 添加一个Pipeline，一个Spider可以有多个Pipeline   | spider .addPipeline(new ConsolePipeline())                   |
| setScheduler(Scheduler)   | 设置Scheduler，一个Spider只能有个一个Scheduler   | spider.setScheduler(new RedisScheduler())                    |
| setDownloader(Downloader) | 设置Downloader，一个Spider只能有个一个Downloader | spider .setDownloader(new SeleniumDownloader())              |
| get(String)               | 同步调用，并直接取得结果                         | ResultItems result = spider.get("http://webmagic.io/docs/")  |
| getAll(String…)           | 同步调用，并直接取得一堆结果                     | List<ResultItems> results = spider .getAll("http://webmagic.io/docs/", "http://webmagic.io/xxx") |

### 爬虫配置Site

```java
private Site site = Site.me()
        .setCharset("UTF-8")//编码
        .setSleepTime(1)//抓取间隔时间
        .setTimeOut(1000*10)//超时时间
        .setRetrySleepTime(3000)//重试时间
        .setRetryTimes(3);//重试次数
```

> 站点本身的一些配置信息，例如编码、HTTP头、超时时间、重试策略等、代理等，都可以通过设置Site对象来进行配置。

| ***\*方法\****           | ***\*说明\****                            | ***\*示例\****                                               |
| ------------------------ | ----------------------------------------- | ------------------------------------------------------------ |
| setCharset(String)       | 设置编码                                  | site.setCharset("utf-8")                                     |
| setUserAgent(String)     | 设置UserAgent                             | site.setUserAgent("Spider")                                  |
| setTimeOut(int)          | 设置超时时间，单位是毫秒                  | site.setTimeOut(3000)                                        |
| setRetryTimes(int)       | 设置重试次数                              | site.setRetryTimes(3)                                        |
| setCycleRetryTimes(int)  | 设置循环重试次数                          | site.setCycleRetryTimes(3)                                   |
| addCookie(String,String) | 添加一条cookie                            | site.addCookie("dotcomt_user","code4craft")                  |
| setDomain(String)        | 设置域名，需设置域名后，addCookie才可生效 | site.setDomain("github.com")                                 |
| addHeader(String,String) | 添加一条addHeader                         | site.addHeader("Referer","[https://github.com](https://github.com/)") |
| setHttpProxy(HttpHost)   | 设置Http代理                              | site.setHttpProxy(new HttpHost("127.0.0.1",8080))            |

## 爬虫分类

- 通用网络爬虫             

   	简单的说就是互联网上抓取所有数据。

- 聚焦网络爬虫            

   	 简单的说就是互联网上只抓取某一种数据。

- 增量式网络爬虫          

   	简单的说就是互联网上只抓取刚刚更新的数据。

- Deep Web 网络爬虫   

   	Deep Web 是那些大部分内容不能通过静态链接获取的、隐藏在搜索表单后的，只有用户提交一些关键词才能获得的 Web 页面。

## 案例 

### Scheduler组件

>  	WebMagic内置了几个常用的Scheduler。如果你只是在本地执行规模比较小的爬虫，那么基本无需定制Scheduler，但是了解一下已经提供的几个Scheduler还是有意义的。

|            类             |                             说明                             |                             备注                             |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| DuplicateRemovedScheduler |                  抽象基类，提供一些模板方法                  |                   继承它可以实现自己的功能                   |
|      QueueScheduler       |                  使用内存队列保存待抓取URL                   |                                                              |
|     PriorityScheduler     |            使用带有优先级的内存队列保存待抓取URL             | 耗费内存较QueueScheduler更大，但是当设置了request.priority之后，只能使用PriorityScheduler才可使优先级生效 |
|  FileCacheQueueScheduler  | 使用文件保存抓取URL，可以在关闭程序并下次启动时，从之前抓取到的URL继续抓取 |       需指定路径，会建立.urls.txt和.cursor.txt两个文件       |
|      RedisScheduler       |      使用Redis保存抓取队列，可进行多台机器同时合作抓取       |                     需要安装并启动redis                      |

>  	 	去重部分被单独抽象成了一个接口：DuplicateRemover，从而可以为同一个Scheduler选择不同的去重方式，以适应不同的需要，目前提供了两种去重方式。

|             类              |                           说明                            |
| :-------------------------: | :-------------------------------------------------------: |
|   HashSetDuplicateRemover   |            使用HashSet来进行去重，占用内存较大            |
| BloomFilterDuplicateRemover | 使用BloomFilter来进行去重，占用内存较小，但是可能漏抓页面 |

### 三种去重的方式

- **Hasset**

   	重复的特点去重。优点是容易理解。使用方便。
	
   	==缺点：==占用内存大，性能较低。

- **redis**

   	(本身速度就很快），而且去重不会占用爬虫服务器的资源，可以处理更大数据量的数据爬取。
	
   	==缺点：==需要准备Redis服务器，增加开发和使用成本。

- **BloomFilter**

   	使用布隆过滤器也可以实现去重。优点是占用的内存要比使用HashSet要小的多，也适合大量数据的去重操作。
   	
   	==缺点：==有误判的可能。没有重复可能会判定重复，但是重复数据一定会判定重复。