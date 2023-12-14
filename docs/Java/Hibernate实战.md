---
title: Hibernate实战
---

## 数据库表设计

想要做一个Online Expense Tracker项目，设计3张表

1. 用户表(users)

- 用户id
- 用户名
- 密码
- 电子邮件

```mysql
CREATE TABLE `users` (
	`user_id` INT PRIMARY KEY AUTO_INCREMENT,
 	`username` VARCHAR(20) NOT NULL,
  `password` VARCHAR(20) NOT NULL,
  `email` VARCHAR(40)
);
```

2. 消费类别表

- 类别id
- 用户id
- 类别名称

```mysql
CREATE TABLE expense_tag (
	`tag_id` INT PRIMARY KEY AUTO_INCREMENT,
  `user_id` INT,
  `tag_name` VARCHAR(20) NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```

3. 消费记录表

- 记录id
- 用户id
- 类别id
- 金额
- 日期
- 备注

```mysql
CREATE TABLE expense_records (
	`record_id` INT PRIMARY KEY AUTO_INCREMENT,
  `user_id` INT,
  `tag_id` INT,
  `amount` DECIMAL(10, 2) NOT NULL,
  `date` DATE NOT NULL,
  `note` VARCHAR(255),
  FOREIGN KEY (user_id) REFERENCES users(user_id),
  FOREIGN KEY (tag_id) REFERENCES expense_tag(tag_id)
);
```



临时

```sql
CREATE TABLE expense (
    `id` INT PRIMARY KEY AUTO_INCREMENT,
    `date` VARCHAR(30),
    `description` VARCHAR(30),
    `price` VARCHAR(30),
    `time` VARCHAR(30),
    `title` VARCHAR(30),
    `user_id` INT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
```





## 概要设计

- 用户注册与登陆

- 对消费记录的增删查改
- 时间线统计金额
- Echarts显示消费占比





## 具体问题



## 其他

- 导航栏激活状态
- 