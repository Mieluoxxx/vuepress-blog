---
title: DES加密算法工作原理
date: 2023/10/31 10:59:00
categories:
    - 信息安全
tags: 
    - 密码学
---

## DES算法简介

**DES(Data Encryption Standard)**是目前最为流行的加密算法之一。DES是**对称的**，也就是说它使用**同一个密钥**来加密和解密数据。

DES还是一种分组加密算法，该算法每次处理固定长度的数据段，称之为分组。DES分组的大小是64位，如果加密的数据长度不是64位的倍数，可以按照某种具体的规则来**填充位**。

从本质上来说，DES的安全性依赖于**虚假表象**，从密码学的术语来讲就是依赖于“混乱和扩散”的原则。混乱的目的是为隐藏任何明文同密文、或者密钥之间的关系，而扩散的目的是使明文中的有效位和密钥一起组成尽可能多的密文。两者结合到一起就使得安全性变得相对较高。

![DES框架.png](https://s2.loli.net/2023/10/31/hlumT1K4JcxOqPv.png)

![DES.png](https://s2.loli.net/2023/10/31/1V7z49nyYqUdrRT.png)

## 伪代码

DES：

1. 初始化密钥K
2. 将输入数据进行IP置换，得到L0和R0
3. 对56位的密钥进行压缩置换，得到子密钥
4. 根据轮数，左移子密钥的部分位数
5. 从56位中选出48位作为新的子密钥
6. 将扩展后的R0与子密钥异或，得到中间结果
7. 将中间结果送入S盒进行替代运算
8. 将替代后的32位输出按照P盒进行置换
9. 将置换后的结果与L0进行异或，得到最终密文
10. 重复步骤2-9，直到完成所有轮次
11. 将最后一轮的左右两部分合并，进行IP-1末置换，得到最终密文



## 重点

### 初始置换和终止置换

目的是为了打乱文本顺序

根据初始置换和终止置换的置换表打乱位置

### 扩展置换

64bit数据分为左32bit，右32bit；右32bit进行扩展置换成48bit

扩展过程：将右32位分成8组，将每组4bit扩展成6bit

（循环，将$S_{i+1}$组的第一位复制给$S_i$组的最后一位；$S_i$组的第一位是$S_{i-1}$组的最后一位）

### S盒压缩处理

经过扩展的48位明文和48位密钥进行异或运算后再使用8个S盒进行压缩处理得到32位数据

取每一组头尾数据，转换为十进制得到行数

取中间数据转换为十进制，得到列数

在压缩表找到对应的压缩数，转换为4bit的二进制数

最后将32位输出进行P盒置换输出

