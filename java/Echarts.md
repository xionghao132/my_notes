# Echarts

## 柱形图bar

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>第一个 ECharts 实例</title>
    <!-- 引入 echarts.js -->
    <script src="https://cdn.staticfile.org/echarts/4.3.0/echarts.min.js"></script>
</head>
<body>
    <!-- 为ECharts准备一个具备大小（宽高）的Dom -->
    <div id="main" style="width: 600px;height:400px;"></div>
    <script type="text/javascript">
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('main'));
        // 指定图表的配置项和数据
        var option = {
            title: {
                text: '第一个 ECharts 实例'
            },
            tooltip: {},
            legend: {
                data:['销量']
            },
            xAxis: {
                data: ["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"]
            },
            yAxis: {},
            series: [{
                name: '销量',
                type: 'bar',
                data: [5, 20, 36, 10, 10, 20]
            }]
        };
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
    </script>
</body>
</html>
```

> 重要的配置项如下

- tooltip 鼠标移动上去显示配置信息
- legend 显示上面的小标签
- series    name显示的是配置信息名字，type可以决定图的类型  data是数据

## 饼图pie

```js
series:[{
          name:'数量',
          type:'pie',    
          radius:'60%', 
          data:[
          {value:500,name:'Android'},
          {value:200,name:'IOS'},
          {value:360,name:'PC'},
          {value:100,name:'Ohter'}
                ]
        }]
```

> 重要配置项如下

- roseType: 'angle',    显示的饼图就不是圆形

- 设置背景阴影

  ```js
   itemStyle: {
                  normal: {
                      shadowBlur: 200,
                      shadowColor: 'rgba(0, 0, 0, 0.5)'
                  }
              }
  ```

## 样式设置

> ECharts 可以通过样式设置来改变图形元素或者文字的颜色、明暗、大小等。

```js
var chart = echarts.init(dom, 'light');
var chart = echarts.init(dom, 'dark');
```

[echarts]: http://echarts.baidu.com/theme-builder/&quot;	"主题编辑器"

 	目前主题下载提供了 JS 版本和 JSON 版本。如果你使用 JS 版本，可以将 JS 主题代码保存一个 **主题名.js** 文件，然后在 HTML 中引用该文件，最后在代码中使用该主题。比如上图中我们选中了一个主题，将 JS 代码保存为 **wonderland.js**。

```js
<!-- 引入主题 -->
<script src="https://www.runoob.com/static/js/wonderland.js"></script>
// HTML 引入 wonderland.js 文件后，在初始化的时候调用
var myChart = echarts.init(dom, 'wonderland');
```

`实例`

```js
// 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('main'), 'wonderland');
        myChart.setOption({
            series : [
                {
                    name: '访问来源',
                    type: 'pie',    // 设置图表类型为饼图
                    radius: '55%',  // 饼图的半径，外半径为可视区尺寸（容器高宽中较小一项）的 55% 长度。
                    data:[          // 数据数组，name 为数据项名称，value 为数据项值
                        {value:235, name:'视频广告'},
                        {value:274, name:'联盟广告'},
                        {value:310, name:'邮件营销'},
                        {value:335, name:'直接访问'},
                        {value:400, name:'搜索引擎'}
                    ]
                }
            ]
        })
```

 	调色盘可以在 option 中设置。调色盘给定了一组颜色，图形、系列会自动从其中选择颜色。可以设置全局的调色盘，也可以设置系列自己专属的调色盘。

```js
option = {
    // 全局调色盘。
    color: ['#c23531','#2f4554', '#61a0a8', '#d48265', '#91c7ae','#749f83',  '#ca8622', '#bda29a','#6e7074', '#546570', '#c4ccd3'],

    series: [{
        type: 'bar',
        // 此系列自己的调色盘。
        color: ['#dd6b66','#759aa0','#e69d87','#8dc1a9','#ea7e53','#eedd78','#73a373','#73b9bc','#7289ab', '#91ca8c','#f49f42'],
        ...
    }, {
        type: 'pie',
        // 此系列自己的调色盘。
        color: ['#37A2DA', '#32C5E9', '#67E0E3', '#9FE6B8', '#FFDB5C','#ff9f7f', '#fb7293', '#E062AE', '#E690D1', '#e7bcf3', '#9d96f5', '#8378EA', '#96BFFF'],
        ...
    }]
}
```

 	在鼠标悬浮到图形元素上时，一般会出现高亮的样式。默认情况下，高亮的样式是根据普通样式自动生成的。

如果要自定义高亮样式可以通过 emphasis 属性来定制：

```js
// 高亮样式。
        emphasis: {
            itemStyle: {
                // 高亮时点的颜色
                color: 'red'
            },
            label: {
                show: true,
                // 高亮时标签的文字
                formatter: '高亮时显示的标签内容'
            }
        },
```

## 异步加载数据

 	ECharts 通常数据设置在 setOption 中，如果我们需要异步加载数据，可以配合 jQuery等工具，在异步获取数据后通过 setOption 填入数据和配置项就行。

 	ECharts 通常数据设置在 setOption 中，如果我们需要异步加载数据，可以配合 jQuery等工具，在异步获取数据后通过 setOption 填入数据和配置项就行。 

`json 数据：`

```json
{
    "data_pie" : [
    {"value":235, "name":"视频广告"},
    {"value":274, "name":"联盟广告"},
    {"value":310, "name":"邮件营销"},
    {"value":335, "name":"直接访问"},
    {"value":400, "name":"搜索引擎"}
    ]
}
```

`实例`

```js
var myChart = echarts.init(document.getElementById('main'));
$.get('https://www.runoob.com/static/js/echarts_test_data.json', function (data) {
    myChart.setOption({
        series : [
            {
                name: '访问来源',
                type: 'pie',    // 设置图表类型为饼图
                radius: '55%',  // 饼图的半径，外半径为可视区尺寸（容器高宽中较小一项）的 55% 长度。
                data:data.data_pie
            }
        ]
    })
}, 'json')
```

>   	如果异步加载需要一段时间，我们可以添加 loading 效果，ECharts 默认有提供了一个简单的加载动画。只需要调用 showLoading 方法显示。数据加载完成后再调用 hideLoading 方法隐藏加载动画：

```js
var myChart = echarts.init(document.getElementById('main'));
myChart.showLoading();  // 开启 loading 效果
$.get('https://www.runoob.com/static/js/echarts_test_data.json', function (data) {
    myChart.hideLoading();  // 隐藏 loading 效果
    myChart.setOption({
        series : [
            {
                name: '访问来源',
                type: 'pie',    // 设置图表类型为饼图
                radius: '55%',  // 饼图的半径，外半径为可视区尺寸（容器高宽中较小一项）的 55% 长度。
                data:data.data_pie
            }
        ]
    })
}, 'json')
```

 	ECharts 使用 dataset 管理数据。dataset 组件用于单独的数据集声明，从而数据可以单独管理，被多个组件复用，并且可以基于数据指定数据到视觉的映射。

 	下面是一个最简单的 dataset 的例子：

```js
  <script type="text/javascript">
        // 基于准备好的dom，初始化echarts实例
        var myChart = echarts.init(document.getElementById('main'));
 
        // 指定图表的配置项和数据
        var option = {
            legend: {},
            tooltip: {},
            dataset: {
                // 提供一份数据。
                source: [
                    ['product', '2015', '2016', '2017'],
                    ['Matcha Latte', 43.3, 85.8, 93.7],
                    ['Milk Tea', 83.1, 73.4, 55.1],
                    ['Cheese Cocoa', 86.4, 65.2, 82.5],
                    ['Walnut Brownie', 72.4, 53.9, 39.1]
                ]
            },
            // 声明一个 X 轴，类目轴（category）。默认情况下，类目轴对应到 dataset 第一列。
            xAxis: {type: 'category'},
            // 声明一个 Y 轴，数值轴。
            yAxis: {},
            // 声明多个 bar 系列，默认情况下，每个系列会自动对应到 dataset 的每一列。
            series: [
                {type: 'bar'},
                {type: 'bar'},
                {type: 'bar'}
            ]
        };
 
        // 使用刚指定的配置项和数据显示图表。
        myChart.setOption(option);
    </script>
```

- source是数据，第一列对应的是x轴，第一个个字段是名称，后面的是有多少个条形统计图
- series里面的类型数量对应每一块条形统计图的项目个数

> 或者使用对象数组的方式

```js
option = {
    legend: {},
    tooltip: {},
    dataset: {
        // 这里指定了维度名的顺序，从而可以利用默认的维度到坐标轴的映射。
        // 如果不指定 dimensions，也可以通过指定 series.encode 完成映射，参见后文。
        dimensions: ['product', '2015', '2016', '2017'],
        source: [
            {product: 'Matcha Latte', '2015': 43.3, '2016': 85.8, '2017': 93.7},
            {product: 'Milk Tea', '2015': 83.1, '2016': 73.4, '2017': 55.1},
            {product: 'Cheese Cocoa', '2015': 86.4, '2016': 65.2, '2017': 82.5},
            {product: 'Walnut Brownie', '2015': 72.4, '2016': 53.9, '2017': 39.1}
        ]
    },
    xAxis: {type: 'category'},
    yAxis: {},
    series: [
        {type: 'bar'},
        {type: 'bar'},
        {type: 'bar'}
    ]
};
```

