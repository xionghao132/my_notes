# Plotly

## 概述

Plotly是一个数据可视化和数据分析的开源Python库。它提供了各种绘图类型，如线图、散点图、条形图、箱型图、热力图等，具有交互性和可定制性。它还提供了一个在线编辑器，可以在web上创建、分享和发布交互式图形。使用Plotly，用户可以快速轻松地制作出漂亮、高质量的可视化图表。

**类型**

- 散点图（Scatter plot）
- 折线图（Line plot）
- 条形图（Bar chart）
- 面积图（Area chart）
- 直方图（Histogram）
- 箱型图（Box plot）
- 热力图（Heatmap）
- 等高线图（Contour plot）
- 3D散点图（3D Scatter plot）
- 3D表面图（3D Surface plot）



## 使用

### 散点图

```Python
import plotly.express as px
import pandas as pd

# 创建示例数据
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [5, 4, 3, 2, 1]
})

# 使用 plotly.express 绘制散点图
fig = px.scatter(data, x='x', y='y', title='Scatter plot')
#fig.show()
fig.write_image("scatter_plot.png")


import plotly.graph_objs as go
import numpy as np

# 创建随机数据
np.random.seed(123)
x = np.random.randn(100)
y = np.random.randn(100)

# 创建散点图
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))

# 设置图表标题和轴标签
fig.update_layout(title='Random Scatter Plot',
                  xaxis_title='X Axis',
                  yaxis_title='Y Axis')

# 显示图表
fig.show()

```



### 折线图

* 普通折线图

```Python
import plotly.graph_objs as go

x = [1, 2, 3, 4, 5]
y = [2, 1, 3, 2.5, 4]

trace = go.Scatter(x=x, y=y, mode='lines+markers')
data = [trace]

layout = go.Layout(title='My Line Chart')

fig = go.Figure(data=data, layout=layout)
fig.show()

```



* 多条折线

```
import plotly.graph_objs as go

x_data = [1, 2, 3, 4, 5]
y1_data = [1, 4, 9, 16, 25]
y2_data = [1, 2, 3, 4, 5]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_data, y=y1_data, mode='lines', name='line1'))
fig.add_trace(go.Scatter(x=x_data, y=y2_data, mode='lines', name='line2'))

fig.show()

```



* 带有子图



```
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

x_data = np.arange(0, 2*np.pi, 0.1)
y1_data = np.sin(x_data)
y2_data = np.cos(x_data)

fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=x_data, y=y1_data, mode='lines', name='sin(x)'), row=1, col=1)
fig.add_trace(go.Scatter(x=x_data, y=y2_data, mode='lines', name='cos(x)'), row=2, col=1)

fig.show()

```



**参数**

- x, y：指定 x 和 y 轴所对应的数据列。
- mode：指定折线图的类型，可以设置为 ‘lines’、‘markers’、‘lines+markers’ 等，其中 ‘lines’
- 表示仅显示线段，‘markers’ 表示仅显示散点，‘lines+markers’ 表示同时显示线段和散点。
- line：一个字典，用于设置折线的属性，包括颜色、宽度、类型等。
- marker：一个字典，用于设置散点的属性，包括颜色、大小、类型等。
- text：用于为每个点添加文本标签，可以是一个字符串数组，也可以是一个数据列。
- hover_name：用于将鼠标悬停在点上时显示的标签指定为数据帧中的列名。
- hover_data：用于将鼠标悬停在点上时显示的数据指定为数据帧中的列名。
- name：为每个数据集指定一个名称，用于生成图例。



### 条形图

```Python
import plotly.express as px

data = {
    'fruit': ['apple', 'banana', 'orange', 'kiwi'],
    'count': [3, 2, 4, 1]
}

fig = px.bar(data, x='fruit', y='count')
fig.show()

```



[【Python】Plotly：最强的Python可视化包（超详细讲解+各类源代码案例）（一）_python plotly-CSDN博客](https://blog.csdn.net/wzk4869/article/details/129864811)