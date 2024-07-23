# Flask

## 概述

Flask是一个用Python编写的Web应用程序框架。

安装：

```python
pip install flask
```

## 目录结构

选择`pycharm`中的`NewProject`，选择`flask`即可

```
| - projectName
	| - app  //程序包
		| - templates //jinjia2模板
		|- static //css,js 图片等静态文件
		| - main  //py程序包 ，可以有多个这种包，每个对应不同的功能
			| - __init__.py
			|- errors.py
			|- forms.py
			|- views.py
		|- __init__.py
		|- email.py //邮件处理程序
		|- models.py //数据库模型
	|- migrations //数据迁移文件夹
	| - tests  //单元测试
		|- __init__.py
		|- test*.py //单元测试程序，可以包含多个对应不同的功能点测试
	|- venv  //虚拟环境
	|- requirements.txt //列出了所有依赖包以及版本号，方便在其他位置生成相同的虚拟环境以及依赖
	|- config.py //全局配置文件，配置全局变量
	|- manage.py //启动程序
```

## 应用

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')  #route()函数是一个装饰器，它告诉应用程序哪个URL应该调用相关的函数
def hello_world():
   return 'Hello World'

if __name__ == '__main__':
   app.run()               #app.run(host, port, debug, options) debug=True提供调试信息
```

## 变量规则

```python
@app.route('/hello/<name>')
def hello_name(name):
   return 'Hello %s!' % name

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo
```

## URL构造规则

```python
@app.route('/user/<name>')
def hello_user(name):
   if name =='admin':
      return redirect(url_for('hello_admin'))
   else:
      return redirect(url_for('hello_guest', guest = name))  #这个guest是参数名
```

## HTTP方法

默认情况下，Flask路由响应**GET**请求。但是，可以通过为**route()**装饰器提供方法参数来更改此首选项。

```python
from flask import Flask, redirect, url_for, request, render_template
@app.route('/')
def index():
    return render_template("login.html")

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      print(1)
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      print(2)
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))
```

## 模板

```python
@app.route('/')
def index():
    # 往模板中传入的数据
    my_str = 'Hello Word'
    my_int = 10
    my_array = [3, 4, 2, 1, 7, 9]
    my_dict = {
        'name': 'xiaoming',
        'age': 18
    }
    return render_template('hello.html',    #传数据 在html中可以直接解析
                           my_str=my_str,
                           my_int=my_int,
                           my_array=my_array,
                           my_dict=my_dict
                           )
```

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Title</title>
</head>
<body>
  我的模板html内容
  <br />{{ my_str }}
  <br />{{ my_int }}
  <br />{{ my_array }}
  <br />{{ my_dict }}
</body>
</html>
```

## 静态文件

```html
<html>

   <head>
      <script type = "text/javascript" 
         src = "{{ url_for('static', filename = 'hello.js') }}" ></script>  <!--使用static目录下的资源-->
   </head>
   
   <body>
      <input type = "button" onclick = "sayHello()" value = "Say Hello" />
   </body>
   
</html>
```

## Request对象

来自客户端网页的数据作为全局请求对象发送到服务器。为了处理请求数据，应该从Flask模块导入。

Request对象的重要属性如下所列：

- **Form** - 它是一个字典对象，包含表单参数及其值的键和值对。
- **args** - 解析查询字符串的内容，它是问号（？）之后的URL的一部分。
- **Cookies** - 保存Cookie名称和值的字典对象。
- **files** - 与上传文件有关的数据。
- **method** - 当前请求方法。

[(44条消息) Flask连接数据库mysql_秀玉轩晨的博客-CSDN博客_flask mysql](https://blog.csdn.net/qq_40552152/article/details/121196396)

[Flask 教程_w3cschool](https://www.w3cschool.cn/flask/)