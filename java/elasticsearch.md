[TOC]

# ElasticSearch

## javaAPI的操作

### 建立环境

 	Elasticsearch 软件是由 Java 语言开发的，所以也可以通过 Java API 的方式对 Elasticsearch

服务进行访问。

> 修改 pom 文件，增加 Maven 依赖关系

```xml
<dependencies>
     <dependency>
         <groupId>org.elasticsearch</groupId>
         <artifactId>elasticsearch</artifactId>
         <version>7.8.0</version>
     </dependency>
     <!-- elasticsearch 的客户端 -->
     <dependency>
         <groupId>org.elasticsearch.client</groupId>
         <artifactId>elasticsearch-rest-high-level-client</artifactId>
         <version>7.8.0</version>
     </dependency>
     <!-- elasticsearch 依赖 2.x 的 log4j -->
     <dependency>
         <groupId>org.apache.logging.log4j</groupId>
         <artifactId>log4j-api</artifactId>
         <version>2.8.2</version>
     </dependency>
     <dependency>
         <groupId>org.apache.logging.log4j</groupId>
         <artifactId>log4j-core</artifactId>
         <version>2.8.2</version>
    </dependency>
        <dependency>
        <groupId>com.fasterxml.jackson.core</groupId>
        <artifactId>jackson-databind</artifactId>
        <version>2.9.9</version>
    </dependency>
     <!-- junit 单元测试 -->
     <dependency>
         <groupId>junit</groupId>
         <artifactId>junit</artifactId>
         <version>4.12</version>
     </dependency>
</dependencies>
```

### 创建客户端对象

 	==注意：==采用高级的REST客户端对象，早期版本的对象在未来会被删除

```java
// 创建客户端对象
RestHighLevelClient client = new RestHighLevelClient(
RestClient.builder(new HttpHost("localhost", 9200, "http"))
);
// 关闭客户端连接
client.close();
```

### 索引操作

#### 创建索引

 	==注意：==将close()放在最后面

```
// 创建索引 - 请求对象
CreateIndexRequest request = new CreateIndexRequest("user");
// 发送请求，获取响应
CreateIndexResponse response = client.indices().create(request, 
RequestOptions.DEFAULT);
boolean acknowledged = response.isAcknowledged();
// 响应状态
System.out.println("操作状态 = " + acknowledged);
```

#### 查看索引

```java
// 查询索引 - 请求对象
GetIndexRequest request = new GetIndexRequest("user");
// 发送请求，获取响应
GetIndexResponse response = client.indices().get(request, 
RequestOptions.DEFAULT);
System.out.println("aliases:"+response.getAliases());
System.out.println("mappings:"+response.getMappings());
System.out.println("settings:"+response.getSettings());
```

#### 删除索引

```java
// 删除索引 - 请求对象
DeleteIndexRequest request = new DeleteIndexRequest("user");
// 发送请求，获取响应
AcknowledgedResponse response = client.indices().delete(request, 
RequestOptions.DEFAULT);
// 操作结果
System.out.println("操作结果 ： " + response.isAcknowledged());
```

### 文档操作

#### 新增文档

```java
@data
class User { 
 private String name; 
 private Integer age; 
 private String sex;}
```

 	创建数据添加到文档中

```java
// 新增文档 - 请求对象
IndexRequest request = new IndexRequest();
// 设置索引及唯一性标识
request.index("user").id("1001");
// 创建数据对象
User user = new User();
user.setName("zhangsan");
user.setAge(30);
user.setSex("男");
ObjectMapper objectMapper = new ObjectMapper();
String productJson = objectMapper.writeValueAsString(user);
// 添加文档数据，数据格式为 JSON 格式
request.source(productJson,XContentType.JSON);
// 客户端发送请求，获取响应对象
IndexResponse response = client.index(request, RequestOptions.DEFAULT);
////3.打印结果信息
System.out.println("_index:" + response.getIndex());
System.out.println("_id:" + response.getId());
System.out.println("_result:" + response.getResult());
```

#### 修改文档

```java
// 修改文档 - 请求对象
UpdateRequest request = new UpdateRequest();
// 配置修改参数
request.index("user").id("1001");
// 设置请求体，对数据进行修改
request.doc(XContentType.JSON, "sex", "女");
// 客户端发送请求，获取响应对象
UpdateResponse response = client.update(request, RequestOptions.DEFAULT);
System.out.println("_index:" + response.getIndex());
System.out.println("_id:" + response.getId());
System.out.println("_result:" + response.getResult())
```

#### 查询文档

```java
//1.创建请求对象
GetRequest request = new GetRequest().index("user").id("1001");
//2.客户端发送请求，获取响应对象
GetResponse response = client.get(request, RequestOptions.DEFAULT);
////3.打印结果信息
System.out.println("_index:" + response.getIndex());
System.out.println("_type:" + response.getType());
System.out.println("_id:" + response.getId());
System.out.println("source:" + response.getSourceAsString());
```

#### 删除文档

```java
//创建请求对象
DeleteRequest request = new DeleteRequest().index("user").id("1");
//客户端发送请求，获取响应对象
DeleteResponse response = client.delete(request, RequestOptions.DEFAULT);
//打印信息
System.out.println(response.toString());
```

#### 批量操作

```java
//创建批量新增请求对象
BulkRequest request = new BulkRequest();
request.add(new 
IndexRequest().index("user").id("1001").source(XContentType.JSON, "name", 
"zhangsan"));
request.add(newIndexRequest().index("user").id("1002").source(XContentType.JSON, "name", 
"lisi"));
request.add(new 
IndexRequest().index("user").id("1003").source(XContentType.JSON, "name", 
"wangwu"));
//客户端发送请求，获取响应对象
BulkResponse responses = client.bulk(request, RequestOptions.DEFAULT);
//打印结果信息
System.out.println("took:" + responses.getTook());
System.out.println("items:" + responses.getItems());
```

#### 批量删除

```java
//创建批量删除请求对象
BulkRequest request = new BulkRequest();
request.add(new DeleteRequest().index("user").id("1001"));
request.add(new DeleteRequest().index("user").id("1002"));
request.add(new DeleteRequest().index("user").id("1003"));
//客户端发送请求，获取响应对象
BulkResponse responses = client.bulk(request, RequestOptions.DEFAULT);
//打印结果信息
System.out.println("took:" + responses.getTook());
System.out.println("items:" + responses.getItems());
```

### 批量操作

#### 批量增加

```java
// 批量插入数据
    BulkRequest request = new BulkRequest();
    request.add(new IndexRequest().index("user").id("1001").source(XContentType.JSON, "name", "zhangsan", "age",30,"sex","男"));
    request.add(new IndexRequest().index("user").id("1002").source(XContentType.JSON, "name", "lisi", "age",30,"sex","女"));
    request.add(new IndexRequest().index("user").id("1003").source(XContentType.JSON, "name", "wangwu", "age",40,"sex","男"));
    BulkResponse response = esClient.bulk(request, RequestOptions.DEFAULT);
    System.out.println(response.getTook());
    System.out.println(response.getItems());
```

#### 批量删除

```java
// 批量删除数据
    BulkRequest request = new BulkRequest();
    request.add(new DeleteRequest().index("user").id("1001"));
    request.add(new DeleteRequest().index("user").id("1002"));
    request.add(new DeleteRequest().index("user").id("1003"));
    BulkResponse response = esClient.bulk(request, RequestOptions.DEFAULT);
    System.out.println(response.getTook());
    System.out.println(response.getItems());
```

### 高级查询

#### 查询索引中的全部数据

```java
// 1. 查询索引中全部的数据
    SearchRequest request = new SearchRequest();
    request.indices("user");
    request.source(new SearchSourceBuilder().query(QueryBuilders.matchAllQuery()));
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);
 	//响应，返回有多少调数据满足条件
	SearchHits hits = response.getHits();
    System.out.println(hits.getTotalHits());
    System.out.println(response.getTook());
    for ( SearchHit hit : hits ) {  //输出所有的数据
        System.out.println(hit.getSourceAsString());
    }
```

#### term 查询，查询条件为关键字

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");
    request.source(new SearchSourceBuilder().query(QueryBuilders.termQuery("age", 30)));
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);
    //响应
```

#### 分页查询

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");
    SearchSourceBuilder builder = new SearchSourceBuilder().query(QueryBuilders.matchAllQuery());
    // (当前页码-1)*每页显示数据条数
    builder.from(2);       //from表示从第几行开始
    builder.size(2);       //每页的数量
    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);
    //响应
```

#### 查询排序

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");
    SearchSourceBuilder builder = new SearchSourceBuilder().query(QueryBuilders.matchAllQuery());
    builder.sort("age", SortOrder.DESC);
    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);
    //响应
```

#### 过滤字段

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");
    SearchSourceBuilder builder = new SearchSourceBuilder().query(QueryBuilders.matchAllQuery());
    String[] excludes = {"age"};
    String[] includes = {};
    builder.fetchSource(includes, excludes);
    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);
   //响应
```

#### 组合查询

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");
    SearchSourceBuilder builder = new SearchSourceBuilder();
    BoolQueryBuilder boolQueryBuilder = QueryBuilders.boolQuery();
    //boolQueryBuilder.must(QueryBuilders.matchQuery("age", 30));//必须满足
    //boolQueryBuilder.must(QueryBuilders.matchQuery("sex", "男"));
    //boolQueryBuilder.mustNot(QueryBuilders.matchQuery("sex", "男"));//必须不满足
	//should类似于or
    boolQueryBuilder.should(QueryBuilders.matchQuery("age", 30));
    boolQueryBuilder.should(QueryBuilders.matchQuery("age", 40));
    builder.query(boolQueryBuilder);
    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);
     //响应
```

#### 范围查询

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");

    SearchSourceBuilder builder = new SearchSourceBuilder();
    RangeQueryBuilder rangeQuery = QueryBuilders.rangeQuery("age");

    rangeQuery.gte(30);
    rangeQuery.lt(50);

    builder.query(rangeQuery);

    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);

     //响应
```

#### 模糊查询

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");

    SearchSourceBuilder builder = new SearchSourceBuilder();     //字符相差两个以内可以查出来
    builder.query(QueryBuilders.fuzzyQuery("name", "wangwu").fuzziness(Fuzziness.TWO));

    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);

    //响应
```

#### 高亮查询

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");

    SearchSourceBuilder builder = new SearchSourceBuilder();
    TermsQueryBuilder termsQueryBuilder = QueryBuilders.termsQuery("name", "zhangsan");

    HighlightBuilder highlightBuilder = new HighlightBuilder();
    highlightBuilder.preTags("<font color='red'>");  //显示的时候能高亮显示
    highlightBuilder.postTags("</font>");
    highlightBuilder.field("name");

    builder.highlighter(highlightBuilder);
    builder.query(termsQueryBuilder);

    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);

    //响应
```

#### 聚合查询

```java
	SearchRequest request = new SearchRequest();
    request.indices("user");

    SearchSourceBuilder builder = new SearchSourceBuilder();
						//AggregationBuilders下面的方法max,min,avg等。里面的名字可以自己命名。
    AggregationBuilder aggregationBuilder = AggregationBuilders.max("maxAge").field("age");
    builder.aggregation(aggregationBuilder);

    request.source(builder);
    SearchResponse response = esClient.search(request, RequestOptions.DEFAULT);

   //响应
```

#### 分组查询

```java
 SearchSourceBuilder builder = new SearchSourceBuilder();
	//terms表示分组
    AggregationBuilder aggregationBuilder = AggregationBuilders.terms("ageGroup").field("age");
    builder.aggregation(aggregationBuilder);
```

## ElasticSearch环境

### Windows集群

> node-100x的配置，先删掉data目录，然后删掉logs里面所有文件

```yml
#节点 1 的配置信息：
#集群名称，节点之间要保持一致
cluster.name: my-elasticsearch
#节点名称，集群内要唯一
node.name: node-1001
node.master: true
node.data: true
#ip 地址
network.host: localhost
#http 端口
http.port: 1001
#tcp 监听端口
transport.tcp.port: 9301  
#discovery.seed_hosts: ["localhost:9301", "localhost:9302","localhost:9303"] #寻找master结点,后面两个结点要开启。
#discovery.zen.fd.ping_timeout: 1m
#discovery.zen.fd.ping_retries: 5
#集群内的可以被选为主节点的节点列表
#cluster.initial_master_nodes: ["node-1", "node-2","node-3"]
#跨域配置
#action.destructive_requires_name: true
http.cors.enabled: true
http.cors.allow-origin: "*"
```

- status ：字段指示看当前集群在总体上是否工作正常。它的三种颜色含义如下:
- green ：所有的主分片和副本分片都正常运行。
- yellow ： 所有的主分片都正常运行，但不是所有的副本分片都正常运行。
- red ： 有主分片没能正常运行。

### linux集群

#### 解压软件到linux

```sh
#在opt/modul/ 下解压
# 解压缩
tar -zxvf elasticsearch-7.8.0-linux-x86_64.tar.gz -C /opt/module
# 改名
mv elasticsearch-7.8.0 es
```

#### 创建用户

 	==注意：==因为安全问题，Elasticsearch 不允许 root 用户直接运行，所以要创建新用户在 root 用户中创建新用户

```sh
useradd es #新增 es 用户
passwd es #为 es 用户设置密码
userdel -r es #如果错了，可以删除再加
chown -R es:es /opt/module/es #文件夹所有者
```

#### 修改 /opt/module/es/config/elasticsearch.yml 文件

```sh
# 加入如下配置
cluster.name: elasticsearch
node.name: node-1
network.host: 0.0.0.0
http.port: 9200
cluster.initial_master_nodes: ["node-1"]
```

#### 修改 /etc/security/limits.conf

```sh
# 在文件末尾中增加下面内容
# 每个进程可以打开的文件数的限制
es soft nofile 65536
es hard nofile 65536
```

#### 修改 /etc/security/limits.d/20-nproc.conf

```sh
# 在文件末尾中增加下面内容
# 每个进程可以打开的文件数的限制
es soft nofile 65536
es hard nofile 65536
# 操作系统级别对每个用户创建的进程数的限制
* hard nproc 4096
# 注：* 带表 Linux 所有用户名称
```

#### 修改 /etc/sysctl.conf

```sh
# 在文件中增加下面内容
# 一个进程可以拥有的 VMA(虚拟内存区域)的数量,默认值为 65536
vm.max_map_count=655360
```

#### 启动软件

> 先到es目录下面，然后切换用户

```
su es
bin/elasticsearch
#穿件文件的时候可能权限不对，执行下面的语句，注意切换到root用户
chown -R es:es /opt/module/es #文件夹所有者
```

## ElasticSearch集成SpringData框架

> 目前最新 springboot 对应 Elasticsearch7.6.2，Spring boot2.3.x 一般可以兼容 Elasticsearch7.x

### 修改pom.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <artifactId>spring-boot-starter-parent</artifactId>
        <groupId>org.springframework.boot</groupId>
        <version>2.3.6.RELEASE</version>
    </parent>

    <groupId>org.example</groupId>
    <artifactId>elasticsearch</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>8</maven.compiler.source>
        <maven.compiler.target>8</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-elasticsearch</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-test</artifactId>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-test</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-configuration-processor</artifactId>
            <optional>true</optional>
        </dependency>
    </dependencies>
</project>
```

### 增加配置文件application.properties

```properties
# es 服务地址
elasticsearch.host=127.0.0.1
# es 服务端口
elasticsearch.port=9200
# 配置日志级别,开启 debug 日志
logging.level.com.atguigu.es=debug
```

### SpringBoot 主程序

```java
@SpringBootApplication
public class SpringDataElasticSearchMainApplication {
    public static void main(String[] args) {
        SpringApplication.run(SpringDataElasticSearchMainApplication.class,args);
    }
}
```

###  数据实体类以及实体类映射操作

1. type : 字段数据类型 
2.  analyzer : 分词器类型 
3. index : 是否索引(默认:true) 
4.  Keyword : 短语,不进行分词 

```java
@Data
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Document(indexName = "product", shards = 3, replicas = 1)
public class Product {
    @Id   //必须有 id,这里的 id 是全局唯一的标识，等同于 es 中的"_id"
    private Long id;//商品唯一标识
    @Field(type = FieldType.Text)
    private String title;//商品名称
    @Field(type = FieldType.Keyword)
    private String category;//分类名称
    @Field(type = FieldType.Double)
    private Double price;//商品价格
    @Field(type = FieldType.Keyword, index = false)
    private String images;//图片地址
}
```

-  ElasticsearchRestTemplate 是 spring-data-elasticsearch 项目中的一个类，和其他 spring 项目中的 template 类似。 

-  在新版的 spring-data-elasticsearch 中，ElasticsearchRestTemplate 代替了原来的 ElasticsearchTemplate。 

- 原因是 ElasticsearchTemplate 基于 TransportClient，TransportClient 即将在 8.x 以后的版本中移除。所 以，我们推荐使用 ElasticsearchRestTemplate。 

- ElasticsearchRestTemplate 基 于 RestHighLevelClient 客户端的。需要自定义配置类，继承 AbstractElasticsearchConfiguration，并实现 elasticsearchClient()抽象方法，创建 RestHighLevelClient 对 象。

  ```java
  @ConfigurationProperties(prefix = "elasticsearch")
  @Configuration
  @Data
  public class ElasticsearchConfig extends AbstractElasticsearchConfiguration {
      private String host ;
      private Integer port ;
  
      public RestHighLevelClient elasticsearchClient() {
          RestClientBuilder builder = RestClient.builder(new HttpHost(host, port));
          RestHighLevelClient restHighLevelClient = new
                  RestHighLevelClient(builder);
          return restHighLevelClient;
      }
  }
  ```

### DAO 数据访问对象

```java
@Repository
public interface ProductDao extends ElasticsearchRepository<Product,Long> {
}
```

### 索引操作

```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class SpringDataESIndexTest {
    @Autowired
    private ElasticsearchRestTemplate elasticsearchRestTemplate;
    //创建索引并增加映射配置
    @Test
    public void createIndex(){
        //创建索引，系统初始化会自动创建索引
        System.out.println("创建索引");
    }
    @Test
    public void deleteIndex(){
        //创建索引，系统初始化会自动创建索引
        boolean flg = elasticsearchRestTemplate.deleteIndex(Product.class);
        System.out.println("删除索引 = " + flg);
    }
}
```

###  文档操作

```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class SpringDataESProductDaoTest {
    @Autowired
    private ProductDao productDao;
    /**
     * 新增 封装好了 自动转成json
     */
    @Test
    public void save(){
        Product product = new Product();
        product.setId(2L);
        product.setTitle("华为手机");
        product.setCategory("手机");
        product.setPrice(2999.0);
        product.setImages("http://www.atguigu/hw.jpg");
        productDao.save(product);
    }
    //修改
    @Test
    public void update(){
        Product product = new Product();
        product.setId(1L);
        product.setTitle("小米 2 手机");
        product.setCategory("手机");
        product.setPrice(9999.0);
        product.setImages("http://www.atguigu/xm.jpg");
        productDao.save(product);
    }
    //根据 id 查询
    @Test
    public void findById(){
        Product product = productDao.findById(1L).get();
        System.out.println(product);
    }
    //查询所有
    @Test
    public void findAll(){
        Iterable<Product> products = productDao.findAll();
        for (Product product : products) {
            System.out.println(product);
        }
    }
    //删除
    @Test
    public void delete(){
        Product product = new Product();
        product.setId(1L);
        productDao.delete(product);
    }
    //批量新增
    @Test
    public void saveAll(){
        List<Product> productList = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Product product = new Product();
            product.setId(Long.valueOf(i));
            product.setTitle("["+i+"]小米手机");
            product.setCategory("手机");
            product.setPrice(1999.0+i);
            product.setImages("http://www.atguigu/xm.jpg");
            productList.add(product);
        }
        productDao.saveAll(productList);
    }
    //分页查询
    @Test
    public void findByPageable(){
        //设置排序(排序方式，正序还是倒序，排序的 id)
        Sort sort = Sort.by(Sort.Direction.DESC,"id");
        int currentPage=0;//当前页，第一页从 0 开始，1 表示第二页
        int pageSize = 5;//每页显示多少条
        //设置查询分页
        PageRequest pageRequest = PageRequest.of(currentPage, pageSize,sort);
        //分页查询
        Page<Product> productPage = productDao.findAll(pageRequest);
        for (Product Product : productPage.getContent()) {
            System.out.println(Product);
        }
    }
}
```

###  文档搜索

```java
@RunWith(SpringRunner.class)
@SpringBootTest
public class SpringDataESSearchTest {
    @Autowired
    private ProductDao productDao;
    /**
     * term 查询
     * search(termQueryBuilder) 调用搜索方法，参数查询构建器对象
     */
    @Test
    public void termQuery(){
        TermQueryBuilder termQueryBuilder = QueryBuilders.termQuery("category", "手机");
                Iterable<Product> products = productDao.search(termQueryBuilder);
        for (Product product : products) {
            System.out.println(product);
        }
    }
    /**
     * term 查询加分页
     */
    @Test
    public void termQueryByPage(){
        int currentPage= 0 ;
        int pageSize = 5;
        //设置查询分页
        PageRequest pageRequest = PageRequest.of(currentPage, pageSize);
        TermQueryBuilder termQueryBuilder = QueryBuilders.termQuery("category", "手机");
                Iterable<Product> products =
                        productDao.search(termQueryBuilder,pageRequest);
        for (Product product : products) {
            System.out.println(product);
        }
    }
}
```

