# LayUI

## 概述

​	 和 [Bootstrap](https://so.csdn.net/so/search?from=pc_blog_highlight&q=Bootstrap) 有些相似，但该框架有个极大的好处就是定义了很多前后端交互的样式接口，如分页表格，只需在前端配置好接口，后端则按照定义好的接口规则返回数据，即可完成页面的展示，极大减少了后端人员的开发成本。

![7e28d4dd1f36429892bdbb3c62d38f8d](7e28d4dd1f36429892bdbb3c62d38f8d.png)

- 使用时我们只需引入下述两个文件即可使用

```html
<!-- LayUI的核心CSS文件 -->
<link rel="stylesheet" type="text/css" href="layui-v2.5.6/layui/css/layui.css"/>
<!-- LayUI的核心JS文件（采用模块化引入） --> 
<script src="layui-v2.5.6/layui/layui.js" type="text/javascript" charset="utf-8"></script>
```

- 这是一个基本的入门页面

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
  <title>开始使用 layui</title>
  <!-- LayUI的核心CSS文件 -->
  <link rel="stylesheet" href="./layui/css/layui.css">
</head>
<body>
 
<!-- 你的 HTML 代码 -->
<!-- LayUI的核心JS文件 -->
<script src="./layui/layui.js"></script>
<script>
    layui.use(['layer', 'form'], function(){
      var layer = layui.layer,
      	  form = layui.form;

      layer.msg('Hello World');
    });
</script> 
</body>
</html>
```

## 布局

## 页面元素

### 按钮

### 主题

向任意 HTML 元素设定 class="layui-btn" ，建立一个基础按钮。通过追加样式为 class="layui-btn-{type}" 来定义其他按钮风格

| 原始 | class="layui-btn layui-btn-primary"  |
| ---- | ------------------------------------ |
| 默认 | class="layui-btn"                    |
| 百搭 | class="layui-btn layui-btn-normal"   |
| 暖色 | class="layui-btn layui-btn-warm"     |
| 警告 | class="layui-btn layui-btn-danger"   |
| 禁用 | class="layui-btn layui-btn-disabled" |

​	![image-20211203225504802](https://gitee.com/HB_XN/picture/raw/master/img/20211208200152.png)
​



### 尺寸

| 大型 | class="layui-btn layui-btn-lg" |
| :--- | :----------------------------- |
| 尺寸 | 组合                           |
| 默认 | class="layui-btn"              |
| 小型 | class="layui-btn layui-btn-sm" |
| 迷你 | class="layui-btn layui-btn-xs" |

![image-20211203225821903](https://gitee.com/HB_XN/picture/raw/master/img/20211208200156.png)

| 尺寸     | 组合                                              |
| :------- | :------------------------------------------------ |
| 大型百搭 | class="layui-btn layui-btn-lg layui-btn-normal"   |
| 正常暖色 | class="layui-btn layui-btn-warm"                  |
| 小型警告 | class="layui-btn layui-btn-sm layui-btn-danger"   |
| 迷你禁用 | class="layui-btn layui-btn-xs layui-btn-disabled" |

![image-20211203230117597](https://gitee.com/HB_XN/picture/raw/master/img/20211208200158.png)

```html
<button type="button" class="layui-btn layui-btn-fluid">流体按钮（最大化适应）</button>
```

![image-20211203232742879](https://gitee.com/HB_XN/picture/raw/master/img/20211208200201.png)

```html
<div class="layui-btn-group">
  <button type="button" class="layui-btn">增加</button>
  <button type="button" class="layui-btn">编辑</button>
  <button type="button" class="layui-btn">删除</button>
</div>
      
<div class="layui-btn-group">
  <button type="button" class="layui-btn layui-btn-sm">
    <i class="layui-icon">&#xe654;</i>
  </button>
  <button type="button" class="layui-btn layui-btn-sm">
    <i class="layui-icon">&#xe642;</i>
  </button>
  <button type="button" class="layui-btn layui-btn-sm">
    <i class="layui-icon">&#xe640;</i>
  </button>
  <button type="button" class="layui-btn layui-btn-sm">
    <i class="layui-icon">&#xe602;</i>
  </button>
</div>
 
<div class="layui-btn-group">
  <button type="button" class="layui-btn layui-btn-primary layui-btn-sm">
    <i class="layui-icon">&#xe654;</i>
  </button>
  <button type="button" class="layui-btn layui-btn-primary layui-btn-sm">
    <i class="layui-icon">&#xe642;</i>
  </button>
  <button type="button" class="layui-btn layui-btn-primary layui-btn-sm">
    <i class="layui-icon">&#xe640;</i>
  </button>
</div>
```

### 图标

- 对 i 标签 设定 `class="layui-icon"`
- 然后对元素加上图标对应的 `font-class`
- 内置图标一览表：[字体图标 - 页面元素 - Layui (layuiweb.com)](https://www.layuiweb.com/doc/element/icon.html)

```html
<i class="layui-icon layui-icon-face-smile" style="font-size: 30px; color: #1E9FFF;"></i>  
```

### 表单

```html
<form class="layui-form" action="">
    <div class="layui-form-item">
        <label class="layui-form-label">输入框</label>
        <div class="layui-input-block">
            <input type="text" name="title" required  lay-verify="required" placeholder="请输入标题" autocomplete="off" class="layui-input">
        </div>
    </div>
    <div class="layui-form-item">
        <label class="layui-form-label">密码框</label>
        <div class="layui-input-inline">
            <input type="password" name="password" required lay-verify="required" placeholder="请输入密码" autocomplete="off" class="layui-input">
        </div>
        <div class="layui-form-mid layui-word-aux">辅助文字</div>
    </div>
    <div class="layui-form-item">
        <label class="layui-form-label">选择框</label>
        <div class="layui-input-block">
            <select name="city" lay-verify="required">
                <option value="">请选择城市</option>
                <option value="0">北京</option>
                <option value="1">上海</option>
                <option value="2">广州</option>
                <option value="3">深圳</option>
                <option value="4">杭州</option>
            </select>
        </div>
    </div>
    <div class="layui-form-item">
        <label class="layui-form-label">复选框</label>
        <div class="layui-input-block">
            <input type="checkbox" name="like[write]" title="写作">
            <input type="checkbox" name="like[read]" title="阅读" checked>
            <input type="checkbox" name="like[dai]" title="发呆">
        </div>
    </div>
    <div class="layui-form-item">
        <label class="layui-form-label">开关</label>
        <div class="layui-input-block">
            <input type="checkbox" name="switch" lay-skin="switch">
        </div>
    </div>
    <div class="layui-form-item">
        <label class="layui-form-label">单选框</label>
        <div class="layui-input-block">
            <input type="radio" name="sex" value="男" title="男">
            <input type="radio" name="sex" value="女" title="女" checked>
        </div>
    </div>
    <div class="layui-form-item layui-form-text">
        <label class="layui-form-label">文本域</label>
        <div class="layui-input-block">
            <textarea name="desc" placeholder="请输入内容" class="layui-textarea"></textarea>
        </div>
    </div>
    <div class="layui-form-item">
        <div class="layui-input-block">
            <button class="layui-btn layui-btn-lg" lay-submit lay-filter="formDemo">立即提交</button>
            <button type="button" class="layui-btn" id="test1">
                <i class="layui-icon">&#xe67c;</i>上传图片
            </button>
        </div>
    </div>
</form>
<script>
    // 必须要导入form模块，才能保证表单正常渲染
    layui.use('form', function(){
        var form = layui.form;
        //监听提交
        form.on('submit(formDemo)', function(data){// data就是表单中的所有数据
            layer.msg(JSON.stringify(data.field));
            return false;
        });
    });
</script>
```

###  数据表格

```html
<table id="demo" lay-filter="test"></table>
<script>
    // 必须要导入 table模块 layui.use('table',...)
    layui.use('table', function(){
        var table = layui.table;
        // 为表格填充数据
        table.render({
            elem: '#demo'     //绑定标签
            ,height: 312
            ,url: '${pageContext.request.contextPath}/data.jsp' //获取数据
            ,page:true // 开启分页
            ,cols: [[ //表头
                {field:'id', title: 'ID', sort: true}
                ,{field:'username', width:80, title: '用户名'}
                ,{field:'sex', width:80, title: '性别', sort: true}
                ,{field:'city',  title: '城市'} //没定义宽度则占满剩余所有宽度，都不定义则所有列均分
                ,{field:'score',width:80, title: '评分', sort: true}
                ,{field:"right",title:"操作",toolbar: '#barDemo'}
            ]]
        });
    });
</script>
```

> 数据格式如下：
>
> - code: 0代表查询成功， 为1是， 会显示msg中的内容
> - count是为了分页准备的，共有多少条数据

```json
// 格式如下：
{"msg":"no data",
"code":0,
"data":[{"id":1,"username":"shine1","sex":"男","city":"保定","score":100},
{"id":2,"username":"shine2","sex":"女","city":"石家庄","score":100},
{"id":3,"username":"shine3","sex":"男","city":"邢台","score":100}],
"count":100}
```

### 分页参数

```html
<table id="demo" lay-filter="test"></table>
<script>
    // 必须要导入 table模块 layui.use('table',...)
    layui.use('table', function(){
        var table = layui.table;
        // 为表格填充数据
        table.render({
            elem: '#demo'
            ,height: 312
            ,url: '${pageContext.request.contextPath}/data.jsp' //获取数据
            ,page: {limit:1//每页显示1条
                    ,limits:[1,2,3] //可选每页条数
                    ,first: '首页' //首页显示文字，默认显示页号
                    ,last: '尾页'
                    ,prev: '<em>←</em>' //上一页显示内容，默认显示 > <
                    ,next: '<i class="layui-icon layui-icon-next"></i>'
                    ,layout:['prev', 'page', 'next','count','limit','skip','refresh'] //自定义分页布局
                   } //开启分页
            ,cols: [[.....]]
        });
    });
</script>
```

显示工具栏

> 右上角工具按钮 toolbar:true

```html
<script>
        // 必须要导入 table模块 layui.use('table',...)
        layui.use('table', function(){
            var table = layui.table;
            // 为表格填充数据
            table.render({
                elem: '#demo'
                ,height: 312
                ,toolbar:true
                ,url: '${pageContext.request.contextPath}/data.jsp' //获取数据
                ,page: {...} //开启分页
                ,cols: [[...]]
            });
        });
    </script>
```

### 操作按钮

```html
<table id="demo" lay-filter="test"></table>
<script>
    // 必须要导入 table模块 layui.use('table',...)
    layui.use('table', function(){
        var table = layui.table;
        // 为表格填充数据
        table.render({
            elem: '#demo'
            ,height: 312
            ,toolbar:true
            ,url: '${pageContext.request.contextPath}/data.jsp' //获取数据
            ,cols: [[ //表头
                {field:'id', title: 'ID', sort: true}
                ,{field:'username', width:80, title: '用户名'}
                ,{field:'sex', width:80, title: '性别', sort: true}
                ,{field:'city',  title: '城市'} //没定义宽度则占满剩余所有宽度，都不定义则所有列均分
                ,{field:'score',width:80, title: '评分', sort: true}
                ,{field:"right",title:"操作",toolbar: '#barDemo'}
            ]]
        });
    });
</script>
<!-- 如下script可以定义在页面的任何位置 -->
<script type="text/html" id="barDemo">
        <a class="layui-btn layui-btn-xs" lay-event="edit">编辑</a>
        <a class="layui-btn layui-btn-danger layui-btn-xs" lay-event="del">删除</a>
</script>
```

### 操作按钮回调

```html
 // 事件注册
table.on('tool(test)', function(obj){
    var data = obj.data; //获得当前行数据
    //获得 lay-event 对应的值（也可以是表头的 event 参数对应的值）
    var layEvent = obj.event;
    var tr = obj.tr; //获得当前行 tr 的 DOM 对象（如果有的话）
    if(layEvent === 'del'){ //删除
        layer.confirm('真的删除行么', function(index){
            // 向服务端发送删除请求
            // 此处可以发送ajax
            obj.del(); //删除对应行（tr）的DOM结构
            layer.close(index);
        });
    } else if(layEvent === 'edit'){ //编辑
        // 向服务端发送更新请求
        // 同步更新缓存对应的值
        obj.update({
            username: 'shine001',
            city: '北京',
            sex:'女',
            score:99});
    }
});
```

### 导航

> 导航条
>
> - class = “layui-nav” 水平导航条
> - class=“layui-nav layui-tree” 垂直导航条

```html
<ul class="layui-nav" lay-filter="">
    <li class="layui-nav-item"><a href="">最新活动</a></li>
    <li class="layui-nav-item layui-this"><a href="">产品</a></li>
    <li class="layui-nav-item"><a href="">大数据</a></li>
    <li class="layui-nav-item">
        <a href="javascript:;">解决方案</a>
        <dl class="layui-nav-child"> <!-- 二级菜单 -->
            <dd><a href="">移动模块</a></dd>
            <dd><a href="">后台模版</a></dd>
            <dd><a href="">电商平台</a></dd>
        </dl>
    </li>
    <li class="layui-nav-item"><a href="">社区</a></li>
</ul>
<script>
    //注意：导航 依赖 element 模块，否则无法进行功能性操作
    layui.use('element', function(){});
</script>
```

### 动画

|         样式表         | 描述             |
| :--------------------: | ---------------- |
|     layui-anim-up      | 从最底层往上滑入 |
|    layui-anim-upbit    | 微微往上滑入     |
|    layui-anim-scale    | 平滑放大         |
| layui-anim-scaleSpring | 弹簧式放大       |
|   layui-anim-fadein    | 渐现             |
|   layui-anim-fadeout   | 渐隐             |
|   layui-anim-rotate    | 360度旋转        |
| 追加：layui-anim-loop  | 循环动画         |

```html
<!-- 整个div会在页面显示时，以特定动画显示出来 -->
<div class="layui-anim layui-anim-up" style="height: 100px">aa</div>
<!-- 额外添加样式类：layui-anim-loop 使得动画循环运行 -->
<div class="layui-anim layui-anim-rotate layui-anim-loop"
     style="text-align:center;line-height: 100px;margin-left:50px;height: 100px;width:100px">bb</div>
```

## 模块

### layer

#### 弹窗方法

> 弹窗方法layer.msg()

```html
 <script>
     // 导入 layer模块
     layui.use(["layer"],function(){
         var layer = layui.layer;
         layer.msg("hello world!!");
         layer.msg("确定吗？",{btn:["确定！","放弃！"],
                           yes:function(i){layer.close(i);layer.msg("yes!!!")},
                           btn2:function(i){layer.close(i);layer.msg("no!!!")}}
                  );
</script>         
```

> 弹窗方法layer.alert()

```html
<script>
        // 导入 layer模块
        layui.use(["layer"],function(){
            var layer = layui.layer;
            //0-6 7种图标  0:warning  1:success  2:error  3:question  4:lock  5:哭脸  6：笑脸
            layer.alert("alert弹框蓝",
                {title:'alert',icon:6 },
                function(){//点击“确定”按钮时的回调
                    layer.msg("好滴");
                }
            );
</script>        
```

> 弹窗方法layer.confirm()

```html
<script>
    // 导入 layer模块
    layer.confirm('是否要删除信息!', {
                      
                        btn: ['确定', '取消']
                    }, function (index) {
                        //移除元素
                        //无法关闭这个消息框
                        layer.closeAll(index);  //加入这个信息点击确定 会关闭这个消息框
                        layer.msg("删除成功!",{ icon: 1, time: 1000 });
                    }
                        );
</script>    
```

#### 弹窗属性

- type 弹窗类型，可选值 0-4
- title 弹窗标题， 可选值text/array
- content 弹窗内容， 可选值 text/html/dom

```html
<script>
    // 导入 layer模块
    layui.use(["layer"],function(){
        var layer = layui.layer;
        layer.open({
            type:2,// 消息框，没有确定按钮        2是iframe窗口 
            title:["hello","padding-left:5px"], // 标题，及标题样式
            content:"usertemplate", //网址 controller 
            area:[200px,400px]   //框大小
        });
    });
</script>
<div id="testmain" style="display:none;padding:10px; height: 173px; width: 275px;">
    hello world!
</div>
```

> ==注意==有的地方只能使用parent.layert.msg()

### layDate

> 日期选择

```html
<form class="layui-form layui-form-pane" action="" method="post">
    <!-- layui-form-item 一个输入项-->
    <div class="layui-form-item">
        <label class="layui-form-label">生日</label>
        <!-- layui-input-block 输入框会占满除文字外的整行 -->
        <div class="layui-input-block">
            <input readonly id="birth" type="text" name="birth" placeholder="请选择生日日期" autocomplete="off" class="layui-input">
        </div>
    </div>
</form>
<script>
    layui.use(["laydate","form"],function(){
        var laydate = layui.laydate;
        var layer  = layui.layer;
        //执行一个laydate实例
        laydate.render({
            elem: '#birth', //指定元素
            format:'yyyy/MM/dd',
            value:'2012/12/12' //默认值
            // value:new Date() //默认值
            //type:"datetime"
        });
    });
</script>
```

### upload

> 上传按钮

```html
<button type="button" class="layui-btn" id="test1">
  <i class="layui-icon">&#xe67c;</i>上传图片
</button>
 
<script src="layui.js"></script>
<script>
layui.use('upload', function(){
  var upload = layui.upload;
   
  //执行实例
  var uploadInst = upload.render({
    elem: '#test1' //绑定元素
    ,url: '/upload/' //上传到哪个地方
    ,done: function(res){
      //上传完毕回调
    }
    ,error: function(){
      //请求异常回调
    }
  });
});
</script>
```

>  基础参数

```js
var upload = layui.upload; //得到 upload 对象
 
//创建一个上传组件
upload.render({
  elem: '#id'
  ,url: ''
  ,done: function(res, index, upload){ //上传后的回调
  
  } 
  //,accept: 'file' //允许上传的文件类型
  //,size: 50 //最大允许上传的文件大小
  //,……
})
```

> 允许你直接在元素上设定基础参数

```html
【HTML】
<button class="layui-btn test" lay-data="{url: '/a/'}">上传图片</button>
<button class="layui-btn test" lay-data="{url: '/b/', accept: 'file'}">上传文件</button>
 
```

```js
【JS】
upload.render({
  elem: '.test'
  ,done: function(res, index, upload){
    //获取当前触发上传的元素，一般用于 elem 绑定 class 的情况，注意：此乃 layui 2.1.0 新增
    var item = this.item;
  }
})
```

[layui上传文件组件(前后端代码实现) - 挑战者V - 博客园 (cnblogs.com)](https://www.cnblogs.com/youcong/p/11440639.html)

### carousel

> 轮播图

```html
<div class="layui-carousel" id="test1">
    <div carousel-item style="text-align: center;line-height: 280px">
        <div>条目1</div>
        <div>条目2</div>
        <div>条目3</div>
        <div>条目4</div>
        <div>条目5</div>
    </div>
</div>
<script>
    layui.use(['carousel'], function(){
        var carousel = layui.carousel;
        //建造实例
        carousel.render({
            elem: '#test1'
            ,width: '100%' //设置容器宽度
            ,arrow: 'always' //始终显示箭头
        });
    });
</script>
```

