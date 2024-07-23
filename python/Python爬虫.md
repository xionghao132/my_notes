# Python爬虫

> ​	网络爬虫（又称为网页蜘蛛，网络机器人，在FOAF社区中间，更经常的称为网页追逐者），是一种按照一定的规则，自动地抓取万维网信息的程序或者脚本。

## 第三方库介绍

### requests

```python
import requests
r = requests.get('https://www.baidu.com/')

# 返回请求状态码，200即为请求成功
print(r.status_code)

# 返回页面代码
print(r.text)

# 对于特定类型请求，如Ajax请求返回的json数据
#print(r.json())  有问题
```

​	当然对于大部分网站都会需要你表明你的身份，我们一般正常访问网站都会附带一个请求头（***headers***）信息，里面包含了你的浏览器，编码等内容，网站会通过这部分信息来判断你的身份，所以我们一般写爬虫也加上一个**headers**；

```python
# 添加headers
headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit'}
r = requests.get('https://www.baidu.com/', headers=headers)
```

`post`请求

```python
# 添加headers
headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit'}

# post请求
data = {'users': 'abc', 'password': '123'}
r = requests.post('https://www.weibo.com', data=data, headers=headers)
```

很多时候等于需要登录的站点我们可能需要保持一个会话，不然每次请求都先登录一遍效率太低，在`requests`里面一样很简单；

```python
# 保持会话
# 新建一个session对象
sess = requests.session()
# 先完成登录
sess.post('maybe a login url', data=data, headers=headers)
# 然后再在这个会话下去访问其他的网址
sess.get('other urls')
```

### beautifulsoup

​	当我们通safa过`requests`获取到整个页面的html5代码之后，我们还得进一步处理，因为我们需要的往往只是整个页面上的一小部分数据，所以我们需要对页面代码html5解析然后筛选提取出我们想要对数据，这时候`beautifulsoup`便派上用场了。

 	相当于requests去获取页面，而beautifulsoup去解析页面

```python
# 选用lxml解析器来解析 
soup = BeautifulSoup(html, 'lxml')
```

[(7条消息) Python爬虫——爬取网页时出现中文乱码问题_lucky_shi的博客-CSDN博客](https://blog.csdn.net/lucky_shi/article/details/104602013#:~:text=首先，我说一下 Python中文乱码 的原因， Python中文乱码 是由于 Python 在解析,网页时 默认用U... python 爬虫 网页乱码 问题 解决方法)

```python
r.encoding='utf8'
data=soup.select('#article h1') #选择元素
print(data[0].text)
```

