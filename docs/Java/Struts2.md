---
title: Struts2入门
---

## 第一个Struts2项目

### MVC设计模式

MVC设计模式是软件工程中的一种软件架构模式,其把1软件系统分为3个基本部分:控制器(controller),视图(View)和模型(Model)

(1) 控制器 -- 负责转发请求,对请求进行处理

(2) 视图 -- 提供用户交互界面

(3) 模型 -- 功能接口,用于编写程序的业务逻辑功能,数据库访问功能等

编写web.xml文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
<web-app xmlns="http://xmlns.jcp.org/xml/ns/javaee"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://xmlns.jcp.org/xml/ns/javaee http://xmlns.jcp.org/xml/ns/javaee/web-app_4_0.xsd"
         version="4.0">
    <display-name>struts2-learn</display-name>
<!--    配置Strut2核心控制器-->
    <filter>
        <filter-name>struts2</filter-name>
        <filter-class>org.apache.struts2.dispatcher.filter.StrutsPrepareAndExecuteFilter</filter-class>
    </filter>
    <filter-mapping>
        <filter-name>struts2</filter-name>
<!--        拦截所有请求-->
        <url-pattern>/*</url-pattern>
    </filter-mapping>
</web-app>
```

编写HelloStrutsAction

```java
public class HelloStrutsAction extends ActionSupport {
    public String execute() throws Exception {
        return SUCCESS;
    }
}
```

配置struts.xml



