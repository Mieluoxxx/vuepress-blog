---
title: SVM支持向量机
date: 2023/11/16
categories:
  - 机器学习
tags:
  - SVM
---
> SVM 是一个非常优雅的算法，具有完善的数学理论，SVM有三宝: 间隔 对偶 核方法

## 支持向量

### 线性可分
在二维空间上，两类点被一条直线完全分开叫做线性可分。
严格的数学定义是：
$D_0$ 和 $D_1$ 是 $n$ 维欧氏空间中的两个点集。如果存在 $n$ 维向量 $w$ 和实数 $b$，使得所有属于 $D_0$ 的点 $x$ 都有 $wx_i+b>0$ ，而对于所有属于 $D_1$ 的点 $x_j$ 则有 $wx_j+b<0$ ，则我们称 $D_0$ 和 $D_1$ 线性可分。

### ### 最大间隔超平面

从二维扩展到多维空间中时，将 $D_0$ 和 $D_1$ 完全正确地划分开的 $wx+b<0$ 就成了一个超平面。

为了使这个超平面更具鲁棒性，我们会去找最佳超平面，以最大间隔把两类样本分开的超平面，也称之为最大间隔超平面。
- 两类样本分别分割在该超平面的两侧；
- 两侧距离超平面最近的样本点到超平面的距离被最大化了。

### 支持向量
![1700484531230.png](http://pic.moguw.top/i/2023/11/20/655b55b7ea355.png)
样本中距离超平面最近的一些点，这些点叫做支持向量。

### SVM 最优化问题
SVM 想要的就是找到各类样本点到超平面的距离最远，也就是找到最大间隔超平面。任意超平面可以用下面这个线性方程来描述：
$$
w^Tx+b=0
$$
二维空间点 $(x,y)$ 到直线 $Ax+By+C=0$ 的距离公式是：
$$
\frac{|Ax+By+C|}{\sqrt{A^2+B^2}}
$$
扩展到 $n$ 维空间后，点 $x=(x_1,x_2\ldots x_n)$ 到直线 $w^Tx+b=0$ 的距离为：
$$
\frac{|w^Tx+b|}{||w||}
$$
其中 $||w||=\sqrt{w_1^2+\cdots+w_n^2}$
如图所示，根据支持向量的定义我们知道，支持向量到超平面的距离为 $d$，其他点到超平面的距离大于 $d$。

于是我们有这样的一个公式：
$$
\left.\left\{\begin{aligned}&\frac{w^Tx+b}{||w||}\geq d\quad y=1\\&\frac{w^Tx+b}{||w||}\leq-d\quad y=-1\end{aligned}\right.\right.
$$
稍作转化可以得到：
$$
\left.\left\{\begin{aligned}&\frac{w^Tx+b}{||w||d}\geq 1\quad y=1\\&\frac{w^Tx+b}{||w||d}\leq-1\quad y=-1\end{aligned}\right.\right.
$$
$||w||d$是正数，我们暂且令它为 1（之所以令它等于 1，是为了方便推导和优化，且这样做对目标函数的优化没有影响），故：
$$
\left.\left\{\begin{aligned}&w^Tx+b\geq1\quad y=1\\&w^Tx+b\leq-1\quad y=-1\end{aligned}\right.\right.
$$
将两个方程合并，我们可以简写为：
$$y(w^Tx+b)\geq1$$
至此我们就可以得到最大间隔超平面的上下两个超平面：
![1700485342178.png](http://pic.moguw.top/i/2023/11/20/655b58e064f40.png)
每个支持向量到超平面的距离可以写为：
$$
d=\frac{|w^Tx+b|}{||w||}
$$
由上述 $y(w^Tx+b)>1>0$ 可以得到 $y(w^Tx+b)=|w^Tx+b|$，所以我们得到：$d=\frac{y|w^Tx+b|}{||w||}$
最大化这个距离：
$$
max \  2\times  \frac{y|w^Tx+b|}{||w||}
$$
这里乘上 2 倍也是为了后面推导，对目标函数没有影响。刚刚我们得到支持向量 $y|w^Tx+b|=1$ ，所以我们得到：
$$
max \frac{2}{||w||}
$$
再做一个转换：
$$
min\frac12||w||^2
$$
所以得到的最优化问题是：
$$
min\frac12||w||^2 s.t. y_i(w^Tx_i+b)\geq1
$$

## 对偶问题

### 拉格朗日乘数法
#### 等式约束优化问题

本科高等数学学的拉格朗日程数法是等式约束优化问题：
$$
\begin{gathered}\min f(x_1,x_2,\ldots,x_n)\\s.t.\quad h_k(x_1,x_2,\ldots,x_n)=0\quad k=1,2,\ldots,l\end{gathered}
$$

我们令 $L(x,\lambda)=f(x)+\sum_{k=1}^l\lambda_kh_k(x)$ 函数 $L(x,y)$ 称为 Lagrange 函数，参数 $\lambda$ 称为 Lagrange 乘子**没有非负要求**。
利用必要条件找到可能的极值点：
$$
\left.\left\{\begin{aligned}\frac{\partial L}{\partial x_i}&=0&i=1,2,\ldots,n\\\frac{\partial L}{\partial\lambda_k}&=0&k=1,2,\ldots,l\end{aligned}\right.\right.
$$
具体是否为极值点需根据问题本身的具体情况检验。这个方程组称为等式约束的极值必要条件。

等式约束下的 Lagrange 乘数法引入了 $l$ 个 Lagrange 乘子，我们将 $x_i$ 与 $\lambda_k$ 一视同仁，把 $\lambda_k$ 也看作优化变量，共有 $(n+l)$ 个优化变量。

#### 不等式约束优化问题
而我们现在面对的是不等式优化问题，针对这种情况其主要思想是将不等式约束条件转变为等式约束条件，引入松弛变量，将松弛变量也是为优化变量。
![1700488792128.png](http://pic.moguw.top/i/2023/11/20/655b66599da9b.png)
以我们的例子为例：
$$
\begin{aligned}minf(w)&=min\frac12||w||^2\\s.t.\quad g_i(w)&=1-y_i(w^Tx_i+b)\leq0\end{aligned}
$$
我们引入松弛变量 $\alpha_i^2$ 得到 $h_i(w,a_i)=g_i(w)+a_i^2=0$ 。这里加平方主要为了不再引入新的约束条件，如果只引入 $\alpha_i$ 那我们必须要保证 $\alpha_i>0$ 才能保证 $h_i(w,a_i)=0$ ，这不符合我们的意愿。

由此我们将不等式约束转化为了等式约束，并得到 Lagrange 函数：
$$
\begin{aligned}
L(w,\lambda,a)& =f(w)+\sum_{i=1}^n\lambda_ih_i(w)  \\
&=f(w)+\sum_{i=1}^n\lambda_i[g_i(w)+a_i^2]\quad\lambda_i\geq0
\end{aligned}
$$
由等式约束优化问题极值的必要条件对其求解，联立方程：
$$
\left.\left\{\begin{aligned}\frac{\partial L}{\partial w_i}&=\frac{\partial f}{\partial w_i}+\sum_{i=1}^n\lambda_i\frac{\partial g_i}{\partial w_i}=0,\\\frac{\partial L}{\partial a_i}&=2\lambda_ia_i=0,\\\frac{\partial L}{\partial\lambda_i}&=g_i(w)+a_i^2=0,\\\lambda_i&\geq0\end{aligned}\right.\right.
$$

(为什么取 $\lambda_i \geq 0$，可以通过几何性质来解释，有兴趣的同学可以查下 KKT 的证明）。

针对

我们有两种情况：
**情形一**：$\lambda_i=0,a_i\neq0$
由于 $\lambda_i=0$，因此约束条件 $g_i(w)$ 不起作用，且  $g_i(w) < 0$
**情形二**：$\lambda_i\neq0,a_i=0$
此时 $g_i(w)=0$ 且 $\lambda_i>0$，可以理解为约束条件 $g_i(w)$ 起作用了，且 $g_i(w)=0$

综合可得：$\lambda_ig_i(w)=0$ ，且在约束条件起作用时 $\lambda_i>0$ ；约束不起作用时 $\lambda_i=0, g_i(w)=0$

由此方程组转换为：
$$
\left.\left\{\begin{aligned}\frac{\partial L}{\partial w_i}&=\frac{\partial f}{\partial w_i}+\sum_{j=1}^n\lambda_j\frac{\partial g_j}{\partial w_i}=0,\\\lambda_ig_i(w)&=0,\\g_i(w)&\leq0\\\lambda_i&\geq0\end{aligned}\right.\right.
$$以上便是不等式约束优化优化问题的 **KKT(Karush-Kuhn-Tucker) 条件**，$\lambda_i$ 称为 KKT 乘子。
