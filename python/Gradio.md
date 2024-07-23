# Gradio

## 概述

快速搭建AI算法可视化部署



## 安装

```
pip install gradio
```



## 快速入门

```python
import gradio as gr
#输入文本处理程序
def greet(name):
    return "Hello " + name + "!"
#接口创建函数
#fn设置处理函数，inputs设置输入接口组件，outputs设置输出接口组件
#fn,inputs,outputs都是必填函数
demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

可以使用Gradio CLI在重载模式下启动应用程序，这将提供无缝和快速的开发



## 参数

### Interface类以及基础模块

- fn：包装的函数
- inputs：输入组件类型，（例如：“text”、"image）
- ouputs：输出组件类型，（例如：“text”、"image）



- 最常用的基础模块构成。

- - 应用界面：gr.Interface(简易场景), **gr.Blocks(定制化场景)**
  - 输入输出：gr.Image(图像), gr.Textbox(文本框), gr.DataFrame(数据框), gr.Dropdown(下拉选项), gr.Number(数字), gr.Markdown, gr.Files
  - 控制组件：gr.Button(按钮)
  - 布局组件：gr.Tab(标签页), gr.Row(行布局), gr.Column(列布局)



### 自定义输入组件

```python
import gradio as gr
def greet(name):
    return "Hello " + name + "!"
demo = gr.Interface(
    fn=greet,
    # 自定义输入框
    # 具体设置方法查看官方文档
    inputs=gr.Textbox(lines=3, placeholder="Name Here...",label="my input"),
    outputs="text",
)
demo.launch()
```



### Interface.launch()方法返回三个值

- - app，为 Gradio 演示提供支持的 FastAPI 应用程序
  - local_url，本地地址
  - share_url，公共地址，当share=True时生成

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

iface = gr.Interface(
    fn=greet,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
)
if __name__ == "__main__":
    app, local_url, share_url = iface.launch()
```



### 多个输入和输出

对于复杂程序，输入列表中的每个组件按顺序对应于函数的一个参数。输出列表中的每个组件按顺序排列对应于函数返回的一个值。

```
import gradio as gr
#该函数有3个输入参数和2个输出参数
def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

demo = gr.Interface(
    fn=greet,
    #按照处理程序设置输入组件
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    #按照处理程序设置输出组件
    outputs=["text", "number"],
)
demo.launch()
```



### 图像组件

Gradio支持许多类型的组件，如image、dataframe、video。

```python
import numpy as np
import gradio as gr
def sepia(input_img):
    #处理图像
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img
#shape设置输入图像大小
demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
demo.launch()
```



## Interface进阶使用

### 全局变量

全局变量的好处就是在调用函数后仍然能够保存，例如在机器学习中通过全局变量从外部加载一个大型模型，并在函数内部使用它，以便每次函数调用都不需要重新加载模型。下面就展示了全局变量使用的好处。

```python
import gradio as gr
scores = []
def track_score(score):
    scores.append(score)
    #返回分数top3
    top_scores = sorted(scores, reverse=True)[:3]
    return top_scores
demo = gr.Interface(
    track_score,
    gr.Number(label="Score"),
    gr.JSON(label="Top Scores")
)
demo.launch()
```



### 会话状态

Gradio支持的另一种数据持久性是会话状态，数据在一个页面会话中的多次提交中持久存在。然而，数据不会在你模型的不同用户之间共享。会话状态的典型例子就是聊天机器人，你想访问用户之前提交的信息，但你不能将聊天记录存储在一个全局变量中，因为那样的话，聊天记录会在不同的用户之间乱成一团。注意该状态会在每个页面内的提交中持续存在，但如果您在另一个标签页中加载该演示（或刷新页面），该演示将不会共享聊天历史。

要在会话状态下存储数据，你需要做三件事。

- 在你的函数中传入一个额外的参数，它代表界面的状态。
- 在函数的最后，将状态的更新值作为一个额外的返回值返回。
- 在添加输入和输出时添加state组件。

```python
import random
import gradio as gr
def chat(message, history):
    history = history or []
    message = message.lower()
    if message.startswith("how many"):
        response = random.randint(1, 10)
    elif message.startswith("how"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, response))
    return history, history
#设置一个对话窗
chatbot = gr.Chatbot().style(color_map=("green", "pink"))
demo = gr.Interface(
    chat,
    # 添加state组件
    ["text", "state"],
    [chatbot, "state"],
    # 设置没有保存数据的按钮
    allow_flagging="never",
)
demo.launch()

```



### Interface交互

**实时变化**：在Interface中设置live=True，则输出会跟随输入实时变化。这个时候界面不会有submit按钮，因为不需要手动提交输入。



**流模式**：在许多情形下，我们的输入是实时视频流或者音频流，那么意味这数据不停地发送到后端，这是可以采用streaming模式处理数据。

```
import gradio as gr
import numpy as np
def flip(im):
    return np.flipud(im)
demo = gr.Interface(
    flip,
    gr.Image(source="webcam", streaming=True),
    "image",
    live=True
)
demo.launch()
```



## 应用分享

###  互联网分享

如果运行环境能够连接互联网，在launch函数中设置share参数为True，那么运行程序后。Gradio的服务器会提供XXXXX.gradio.app地址。通过其他设备，比如手机或者笔记本电脑，都可以访问该应用。这种方式下该链接只是本地服务器的代理，不会存储通过本地应用程序发送的任何数据。这个链接在有效期内是免费的，好处就是不需要自己搭建服务器，坏处就是太慢了，毕竟数据经过别人的服务器。

```
demo.launch(share=True)
```



### huggingface托管

为了便于向合作伙伴永久展示我们的模型App,可以将gradio的模型部署到 HuggingFace的 Space托管空间中，完全免费的哦。

方法如下：

1，注册huggingface账号：https://huggingface.co/join

2，在space空间中创建项目：https://huggingface.co/spaces

3，创建好的项目有一个Readme文档，可以根据说明操作，也可以手工编辑app.py和requirements.txt文件。



### 局域网分享

通过设置server_name=‘0.0.0.0’（表示使用本机ip）,server_port（可不改，默认值是7860）。那么可以通过本机ip:端口号在局域网内分享应用。

```
#show_error为True表示在控制台显示错误信息。
demo.launch(server_name='0.0.0.0', server_port=8080, show_error=True)
```

这里host地址可以自行在电脑查询，C:\Windows\System32\drivers\etc\hosts 修改一下即可 127.0.0.1再制定端口号

**密码验证**
在首次打开网页前，可以设置账户密码。比如auth参数为（账户，密码）的元组数据。这种模式下不能够使用queue函数。

```
demo.launch(auth=("admin", "pass1234"))
```


如果想设置更为复杂的账户密码和密码提示，可以通过函数设置校验规则。

#账户和密码相同就可以通过

```
def same_auth(username, password):
    return username == password
demo.launch(auth=same_auth,auth_message="username and password must be the same")
```



原文链接：https://blog.csdn.net/sinat_39620217/article/details/130343655