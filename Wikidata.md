# Wikidata

## 例子

```SPARQL
SELECT ?subject ?predicate ?object
WHERE {
  wd:Q42 ?predicate ?object.
  BIND(wd:Q42 AS ?subject)
   FILTER(LANG(?object) = "en") # Filter by English language label
}
LIMIT 10
```



```SPARQL
SELECT ?pLabel ?oLabel 
WHERE 
{
  wd:Q24 ?p ?o.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
```

可以直接在对应的页面找到对应的URL

这里wd代表了前缀，https://www.wikidata.org/wiki/



```bash
rdfs:label
```

一般使用这个谓语将编号的内容显示出来



1. **`wd:` (Wikidata Item)：** 该前缀用于表示Wikidata上的实体。在SPARQL中，实体通常用 Q 开头的标识符表示。例如，Q42 表示Douglas Adams。使用 `wd:Q42`，您可以引用这个实体。

   ```
   sparqlCopy codeSELECT ?item ?itemLabel
   WHERE {
     wd:Q42 rdfs:label ?itemLabel.
   }
   ```

2. **`wdt:` (Wikidata Truthy)：** 该前缀用于表示属性和实体之间的关系。在SPARQL中，属性通常用 P 开头的标识符表示。例如，P31 表示 "instance of"（是什么的实例）。当您想要检索实体的属性时，可以使用 `wdt:`。例如，`wdt:P31` 表示实体的类型。

   ```
   sparqlCopy codeSELECT ?item ?itemLabel
   WHERE {
     ?item wdt:P31 wd:Q5; # 返回所有类型为人类的实体
           rdfs:label ?itemLabel.
   }
   ```

这两个前缀是为了简化SPARQL查询语法，使其更易读和理解。`wd:` 用于表示实体，而 `wdt:` 用于表示实体与属性之间的关系。

Optional字段 如果存在什么什么属性就查找 



可以直接在浏览器中使用的一个api，目前正在往python上迁移

```text
https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles=Berlin&languages=en&format=json
```

这里有一些api介绍：[MediaWiki API help - Wikidata](https://www.wikidata.org/w/api.php?action=help&modules=main)



```SPARQL
SELECT ?property ?propertyLabel ?propertyDescription (GROUP_CONCAT(DISTINCT(?altLabel); separator = ", ") AS ?altLabel_list) WHERE {
    ?property a wikibase:Property .
    OPTIONAL { ?property skos:altLabel ?altLabel . FILTER (lang(?altLabel) = "en") }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" .}
 }
GROUP BY ?property ?propertyLabel ?propertyDescription
LIMIT 5000
```



[实战 Wikipedia 与 Wikidata 知识图谱数据获取 (8) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/677586518)

这个链接有对一段信息的描述，很实用。



## 使用Python访问

```python
# 设置代理
    proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}
    url = "https://www.wikidata.org/w/api.php"
    params = {
    "action": "wbsearchentities",
    "search": "Fudan",
    "language": "en",
    #"limit": "1",
    "format": "json"
    }
    response = requests.get(url=url, params=params,proxies=proxies)
    result = response.json()
    print(result)
```



还是使用SPARQL靠谱一点。



[Python的Wikipedia API入门 | 码农家园 (codenong.com)](https://www.codenong.com/s-getting-started-with-pythons-wikipedia-api/)

[Wikidata:SPARQL教程 - Wikidata](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial/zh)



[RDF 和 SPARQL 初探：以维基数据为例 - 阮一峰的网络日志 (ruanyifeng.com)](https://www.ruanyifeng.com/blog/2020/02/sparql.html)









