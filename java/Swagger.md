# Swagger

## swagger简介

- 号称世界上最流行的API框架
- Restful Api 文档在线自动生成器 => **API 文档 与API 定义同步更新**
- 直接运行，在线测试API
- 支持多种语言 （如：Java，PHP等）
- 官网：https://swagger.io/

## springboot集成swagger

### maven依赖

```xml
<!-- https://mvnrepository.com/artifact/io.springfox/springfox-swagger2 -->
<dependency>
   <groupId>io.springfox</groupId>
   <artifactId>springfox-swagger2</artifactId>
   <version>2.9.2</version>
</dependency>
<!-- https://mvnrepository.com/artifact/io.springfox/springfox-swagger-ui -->
<!--ui选择了一个版本低一点的 -->
<dependency>
   <groupId>io.springfox</groupId>
   <artifactId>springfox-swagger-ui</artifactId>
   <version>2.8.0</version>
</dependency>
```

### 配置类

```java
package com.config;


import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import springfox.documentation.builders.RequestHandlerSelectors;
import springfox.documentation.service.ApiInfo;
import springfox.documentation.service.Contact;
import springfox.documentation.spi.DocumentationType;
import springfox.documentation.spring.web.plugins.Docket;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

import java.util.ArrayList;

@Configuration
@EnableSwagger2   //开启swagger2
public class SwaggerConfig {
    //配置了swagger的Docket的bean实例
    @Bean
    public Docket docket(){
        return new Docket(DocumentationType.SWAGGER_2)
                .apiInfo(apiInfo())
                .enable(true)   //默认启用    发布可以变成false
                .groupName("熊浩")
                .select()
                //还可以扫描注解 withClassAnnotation 扫描方法上的注解
                .apis(RequestHandlerSelectors.basePackage("com.controller"))      //配置要扫描接口的方式 .any()全扫描 .none（）
                //过滤路径
               // .paths(PathSelectors.ant("/com/**"))
                .build();
    }

    //配置文档信息
    private ApiInfo apiInfo() {
        //作者信息
        Contact contact = new Contact("熊浩", "https://gitee.com/HB_XN", "1329424972@qq.com");
        return new ApiInfo(
                "Swagger学习", // 标题
                "学习演示如何配置Swagger", // 描述
                "v1.0", // 版本
                "https://gitee.com/HB_XN", // 组织链接
                contact, // 联系人信息
                "Apach 2.0 许可", // 许可
                "许可链接", // 许可连接
                new ArrayList<>()// 扩展
        );
    }
    //配置swagger信息apiInfo
}

```

访问测试 http://localhost/swagger-ui.html

### 实体类

```java
@Data   //注意要配置get set方法不然 前端显示不了private 属性
@NoArgsConstructor
@AllArgsConstructor
@ApiModel("用户实体类")   
public class User {
    @ApiModelProperty("用户名")  //放在model的字段上的
    private String username;
    @ApiModelProperty("密码")
    private String password;
}
```

### 常用注解

> @Api(tags=)     放在controller上面，描述controller

> @ApiOperation("接口")  放在方法上面，描述方法

> @ApiParam(“参数说明”)  放在参数旁边，描述参数 ==写在post上 不要紧    写在get上需要加上@RequestParam注解==

### 接口测试

![image-20211227165949756](https://gitee.com/HB_XN/picture/raw/master/img/20211227173645.png)

### 皮肤扩展

Layui-ui  **访问 http://localhost:8080/docs.html**

```xml
<!-- 引入swagger-bootstrap-ui包 /doc.html-->
<dependency>
   <groupId>com.github.xiaoymin</groupId>
   <artifactId>swagger-bootstrap-ui</artifactId>
   <version>1.9.1</version>
</dependency>
```

![image-20211227173312775](https://gitee.com/HB_XN/picture/raw/master/img/20211227173652.png)

