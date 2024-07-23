# SPARQL For Freebase

## SPARQL简介

SPARQL（SPARQL Protocol and RDF Query Language）是一种数据查询语言。它不仅仅支持查询 RDF 数据，也可以在部分关系型数据库中对数据库进行数据操作。

## 符号

### 变量和常量

在 SPARQL 语句中，通常以 `？变量名` 表示变量；而常量一般为字符串、数字及 URI，其中 URI 由尖括号（< >）包裹。

### 标点符号

* `#`是注释符号
* `,` 代表下一个三元组与当前三元组拥有**相同的主语和谓语**。
* `<>`尖括号常用来**包裹 URI**。
* `.`表示`and`条件

## 语句

### 查询

SPARQL 支持多种关键词查询数据。这些关键词包括 `SELECT` 、`CONSTRUCT` 、`DESCRIBE` 、 `ASK` 。

- `SELECT` 查询结果返回一个二维表（与 SQL 中 `SELECT` 类似），其语句一般格式如下：

```SPARQL
SELECT [DISTINCT] <variable1> [<variable2> ...]
[FROM ...]
WHERE
{
    triple pattern 1.
    [triple pattern 2.]
    ...
    [附加条件...]
}
[OFFSET 数字]
[LIMIT 数字]
[ORDER BY | GROUP BY ...]
```

* `CONSTRUCT` 查询结果返回一个 RDF 图（三元组集合），其语句一般格式如下：

```SPARQL
CONSTRUCT 
{ 
    triple pattern .
    ...
} 
WHERE 
{ 
    triple pattern . 
    ...
}
```

* `ASK` 查询结果返回真或者假，表示 RDF 数据中是否存在指定模式的三元组，其语句一般格式如下：

```SPARQL
ASK  
{ 
    triple pattern .
}
```

* `DESCRIBE` 查询结果返回对指定数据的资源描述（以 RDF 图的形式存储），该图的结果由 SPARQL 处理器决定（也就是说，不同 SPARQL 处理器运行同一条 `DESCRIBE` 查询语句，可能会有不同的结果），其语句一般格式如下：

```SPARQL
DESCRIBE <variable1>|<IRI1> [<variable2>|<IRI2> ...]
WHERE 
{
    triple pattern .
}
```

## 关键词

### From

### Filter

## Freebase简介

Freebase 的数据被存储在一个叫做图的数据结构中。一个图由边连接的结点组成。在 Freebase 中，结点使用 /type/object 定义，边使用 /type/link 定义。通过以图的形式存储数据，Freebase 可以快速遍历主题（topic）之间的任意连接，并轻松添加新的模式（schema），而无需改变数据的结构。

## 域和 ID

就像属性被归为类型一样，类型本身也被归为域。把域想象成你最喜欢的报纸上的栏目。 商业，生活方式，艺术和娱乐，政治，经济等。每一个域都有一个 ID（标识符），例如：

- /business 是商业领域的 ID
- /music 音乐领域
- /film 电影领域
- /medicine 医药领域

域的标识符如同文件路径，或 Web 地址的路径。

每个类型也被分配一个标识符，该标识符基于它所属的域。例如，

- /business/company，Company 类型属于 Business 域。
- /music/album
- /film/actor
- /medicine/disease

正如一个类型从它的域继承它的 ID 开头一样，一个属性也从它所属的类型继承它的 ID 开头。例如，公司类型的行业属性（用于指定公司所在的行业）被赋予了 ID /business/company/industry。下面是其他一些例子：

- /automotive/engine/horsepower
- /astronomy/star/planet_s
- /language/human_language/writing_system

因此，即使类型在 Freebase 中没有被安排成层次结构；域、类型和属性在概念上被赋予 ID，以类似文件目录的层次结构来安排

## 复合值类型

复合值类型是 Freebase 中的一种类型，用于表示每个条目由多个字段组成的数据。

考虑这样一个例子，一个城市的人口会随着时间变化，即每次查询 Freebase 中的人口数据时，隐含地是在询问某个日期的人口。这涉及到两个 value，一个是人口数量，一个是日期。在这种情况下，CVT 就非常有效。如果没有 CVT，对人口数据进行建模，需要添加一个主题，将其命名为类似于“1997 年的温哥华人口”的名称，然后在此提交数据。

CVT 可以被认为是一个不需要展示名称的主题，和普通主题一样有一个 GUID，可以被独立引用。然而，Freebase 客户端对 CVT 的处理方式与主题有很大不同。多数情况下，CVT 的每个属性都是非歧义属性。

## 主题的机器标识符 MID

虽然一个主题可能或可能不会用命名空间/密钥 ID 来识别，但它总是可以用 MID，即机器标识符来识别，它由 /m 和一个基数为 32 的唯一标识符组成。MID 在创建时被分配给主题，并在主题的整个生命周期中被管理。

当主题被合并或拆分时，MID 可以发挥关键作用，允许外部应用跟踪逻辑主题，即使物理的 Freebase 标识（主题的 GUID）可能改变。

机器生成的 MID 与其他人类可读的 Freebase ID 的不同之处在于，它们是：

- 保证是存在的
- 由机器产生
- 旨在支持离线比较
- 不是为了向人类传达含义设计的
- 长度较短，可能是固定长度
- 外部系统和组件之间快速交换密钥的理想选择
- MID 是被推荐用于处理 Freebase 主题的标识符

这篇文章对表字段有一些简单介绍

[Freebase Datadump结构初探（一） - 简书 (jianshu.com)](https://www.jianshu.com/p/f03a3664d295)

[Freebase Datadump结构初探（二） - 简书 (jianshu.com)](https://www.jianshu.com/p/157b095d8210)

[(242条消息) 聊天机器人之知识图谱 Freebase 简介_freebase三元组_Chatopera 研发团队的博客-CSDN博客](https://blog.csdn.net/samurais/article/details/108587755)

## Freebase查询模板

* 查询Freebase实体ns:m.054c1的英文名称。

```SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
select distinct ?name 
where {
  ns:m.054c1 ns:type.object.name ?name.
  FILTER(LANGMATCHES(LANG(?name), "en")).
}
```

* 查询具有英文名称"Michael Jordan"@en的Freebase实体。

```SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
select distinct ?entity
where { 
  ?entity ns:type.object.name "Michael Jordan"@en.
}
```

* 查询具有英文名称"Michael Jordan"@en的Freebase实体数量。

```SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
select (count(?entity) as ?cnt) 
where {
  select distinct ?entity
  where { 
    ?entity ns:type.object.name "Michael Jordan"@en.
  }
}
```

## SPARQLWrapper

python操作SPARQL的库

```SPARQL
#导入包
from SPARQLWrapper import SPARQLWrapper, JSON
#创建query和update对象，以便进行各种操作
#XXX是数据库的名字
query = SPARQLWrapper("http://localhost:3030/XXX/query")
update = SPARQLWrapper("http://localhost:3030/XXX/update")
#这两句很关键
query.setReturnFormat(JSON)
update.setMethod('POST')
#远程操作
query.query().convert()
update.query()
```

```SPARQL
update = SPARQLWrapper("http://localhost:3030/xxx/update")
update.setMethod('POST')
#setQuery里面的内容就是原始命令的字符串形式（当然，注释形式也可）
update.setQuery(
"PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"
+ "PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>"
+ "PREFIX ex:   <http://example.org/>"
+ "PREFIX zoo:   <http://example.org/zoo/>"
+ "PREFIX owl:   <http://www.w3.org/2002/07/owl#>"

+ "DELETE DATA {"
+ "ex:dog1    rdf:type         ex:animal ."
+ "ex:cat1    rdf:type         ex:cat ."
+ "ex:cat     rdfs:subClassOf  ex:animal ."
+ "zoo:host   rdfs:range       ex:animal ."
+ "ex:zoo1    zoo:host         ex:cat2 ."
+ "ex:cat3    rdf:sameAs       ex:cat2 ."
+ "}"
        )
update.query()
```

[SPARQL 简易入门 | CosmosNing 的个人博客](https://cosmosning.github.io/2020/07/22/sparql-grammar-tutorial/)

[Freebase 的基本概念 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/354058339)

