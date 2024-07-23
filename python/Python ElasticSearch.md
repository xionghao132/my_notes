# Python ElasticSearch

## 概述

运维里很多操作都离不开日志，而ELK是现在企业里经常使用的日志收集和分析平台，开源，API完善，资源丰富，大家都爱它。elasticsearch，也就是ELK里的"E"，是一个非常强大的搜索和分析引擎，并且提供了Python使用的模块。



## 代码

### 安装

```sh
pip install elasticsearch==7.5.1
```

**注意：**版本匹配问题



### 3连接

```python
from elasticsearch import Elasticsearch
# 连接ES
es = Elasticsearch([{'host':'localhost','port':9200}],http_auth=("username", "secret"),  timeout=3600)
```



### 查询

```python
# 查询
query = {
  "query": {
    "match_all": {}
  }
}
result = es.search(index="megacorp", body=query)
print(result)
```



### 使用DSL语句查询

* `term` 过滤`--term`主要用于精确匹配哪些值，比如数字，日期，布尔值或 `not_analyzed` 的字符串(未经切词的文本数据类型)

```python
query = {
    "query": {
        "term":{
            'age': 32
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
# first_name 可能经过切词了
query = {
    "query": {
        "term":{
            'first_name': 'Jane'
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* `terms` 过滤`--terms` 跟 `term` 有点类似，但 `terms` 允许指定多个匹配条件。 如果某个字段指定了多个值，那么文档需要一起去做匹配

```python
query = {
    "query": {
        "terms":{
            'age': [32, 25]
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
# first_name 可能经过切词了
query = {
    "query": {
        "terms":{
            'first_name': ['Jane','John']
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

- `range` 过滤--按照指定范围查找一批数据

- - `gt` : 大于
  - `gte` : 大于等于
  - `lt` : 小于
  - `lte` : 小于等于

```python
query = {
    "query": {
        "range":{
            'age': {
                "gt":34
            }
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* `exists` 和 `missing` 过滤--查找文档中是否包含指定字段或没有某个字段，类似于`SQL`语句中的`IS_NULL`条件

```python
query = {
    "query": {
        "exists":   {
            "field":    "first_name"
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

- `bool `过滤--合并多个过滤条件查询结果的布尔逻辑

- - `must` :: 多个查询条件的完全匹配,相当于 `and`。
  - `must_not `:: 多个查询条件的相反匹配，相当于 `not`。
  - `should` :: 至少有一个查询条件匹配, 相当于` or`。‘

```python
query = {
    "query": {
         "bool": {
             "must": {
                 "term": { "_score": 1 },
                 "term": { "age": 32 }
                },
             }
         }
}
result = es.search(index="megacorp", body=query)
print(result)
query = {
    "query": {
         "bool": {
             "must": {
                 "term": { "age": 32 }
                },
             "must_not":{
                 "exists":   {
                    "field":    "name"
                }
             }
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* `match_all` 查询--可以查询到所有文档，是没有查询条件下的默认语句。

```python
# 做精确匹配搜索时，你最好用过滤语句，因为过滤语句可以缓存数据。
# match查询只能就指定某个确切字段某个确切的值进行搜索，而你要做的就是为它指定正确的字段名以避免语法错误。
query = {
    "query": {
        "match": {
            "about": "rock"
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* `multi_match `查询--`match`查询的基础上同时搜索多个字段，在多个字段中同时查一个

```python
query = {
    "query": {
        "multi_match": {
            "query": 'music',
             "fields": ["about","interests"]
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* `bool` 查询--与 `bool` 过滤相似，用于合并多个查询子句。不同的是，`bool` 过滤可以直接给出是否匹配成功， 而`bool` 查询要计算每一个查询子句的 `_score `（相关性分值）。

```python
# bool 查询 条件是查询， bool 过滤 条件是过滤
query = {
    "query": {
         "bool": {
             "must": {
                 "match": { "last_name": 'Smith' }
                },
             "must_not":{
                 "exists":   {
                    "field":    "name"
                }
             }
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```



* `wildcards` 查询--使用标准的`shell`通配符查询

```python
query = {
    "query": {
        "wildcard": {
            "about": "ro*"
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* `regexp` 查询

```python
query = {
    "query": {
        "regexp": {
            "about": ".a.*"
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* `prefix` 查询 -- 以什么字符开头的

```python
query = {
    "query": {
        "prefix": {
            "about": "I love"
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

* 短语匹配(Phrase Matching) -- 寻找邻近的几个单词

```python
query = {
    "query": {
        "match_phrase": {
            "about": "I love"
        }
    }
}
result = es.search(index="megacorp", body=query)
print(result)
```

### 统计查询功能

```python
query = {
    "query": {
        "match_phrase": {
            "about": "I love"
        }
    }
}
result = es.count(index="megacorp", body=query)
print(result)
```

### 插入

```python
# 不指定id 自动生成
es.index(index="megacorp",body={"first_name":"xiao","last_name":"xiao", 'age': 25, 'about': 'I love to go rock climbing', 'interests': ['game', 'play']})
{'_index': 'megacorp',
 '_type': '_doc',
 '_id': '3oXEzm4BAZBCZGyZ2R40',
 '_version': 1,
 'result': 'created',
 '_shards': {'total': 2, 'successful': 1, 'failed': 0},
 '_seq_no': 1,
 '_primary_term': 2}
# 指定IDwu
es.index(index="megacorp",id=4,body={"first_name":"xiao","last_name":"wu", 'age': 66, 'about': 'I love to go rock climbing', 'interests': ['sleep', 'eat']})
{'_index': 'megacorp',
 '_type': '_doc',
 '_id': '4',
 '_version': 1,
 'result': 'created',
 '_shards': {'total': 2, 'successful': 1, 'failed': 0},
 '_seq_no': 5,
 '_primary_term': 2}
```

### 删除数据

```python
# 根据ID删除
es.delete(index='megacorp', id='3oXEzm4BAZBCZGyZ2R40')
{'_index': 'megacorp',
 '_type': '_doc',
 '_id': '3oXEzm4BAZBCZGyZ2R40',
 '_version': 2,
 'result': 'deleted',
 '_shards': {'total': 2, 'successful': 1, 'failed': 0},
 '_seq_no': 3,
 '_primary_term': 2}
# delete_by_query：删除满足条件的所有数据，查询条件必须符合DLS格式
query = {
    "query": {
        "match": {
            "first_name": "xiao"
        }
    }
}
result = es.delete_by_query(index="megacorp", body=query)
print(result)
```

### 更新

```python
# 根据ID更新
doc_body = {
    'script': "ctx._source.remove('age')"
}  

# 增加字段   
doc_body = {
    'script': "ctx._source.address = '合肥'"
} 

# 修改部分字段
doc_body = {
    "doc": {"last_name": "xiao"}
}
es.update(index="megacorp", id=4, body=doc_body)
{'_index': 'megacorp',
 '_type': '_doc',
 '_id': '4',
 '_version': 2,
 'result': 'updated',
 '_shards': {'total': 2, 'successful': 1, 'failed': 0},
 '_seq_no': 6,
 '_primary_term': 2}
# update_by_query：更新满足条件的所有数据，写法同上删除和查询
query = {
    "query": {
        "match": {
            "last_name": "xiao"
        }
    },
    "script":{
        "source": "ctx._source.last_name = params.name;ctx._source.age = params.age",
        "lang": "painless",
        "params" : {
            "name" : "wang",
            "age": 100,
        },  
    }

}
result = es.update_by_query(index="megacorp", body=query)
print(result)
```





[python 操作 ElasticSearch 入门 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/95163799)



