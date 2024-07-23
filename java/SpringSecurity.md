# SpringSecurity

## 简介

>  	Spring 是非常流行和成功的 Java 应用开发框架，Spring Security 正是 Spring 家族中的 成员。

>  	Spring Security 基于 Spring 框架，提供了一套 Web 应用安全性的完整解决方 案。

##  SpringSecurity 入门案例

### pom.xml

```xml
<dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
</dependency>
```

### controller

```java
@RestController
@RequestMapping("test")
public class TestController {

    @GetMapping("hello")
    public String hello(){
        return "hello security";
    }
}
```

## 两个重要的接口

**UserDetailsService**   查询数据库用户名和密码过程

- 创建类继承**UsernamePasswordAuthenticationFilter**,重写三个方法
- 创建类实现**UserDetailService**，编写查询数据过程，返回User对象，这个User对象是安全框架提供的对象

**PasswordEncoder** 数据加密接口，用于返回User对象里面密码加密

## Web权限方案

1. 认证
2. 授权

### 设置登录的用户名和密码

- 配置文件
- 配置类
- 自定义实现类

### pom.xml

```xml
		<dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-security</artifactId>
        </dependency>
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>8.0.21</version>
        </dependency>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <version>1.18.8</version>
        </dependency>
        <dependency>
            <groupId>com.baomidou</groupId>
            <artifactId>mybatis-plus-boot-starter</artifactId>
            <version>3.0.5</version>
        </dependency>
```

