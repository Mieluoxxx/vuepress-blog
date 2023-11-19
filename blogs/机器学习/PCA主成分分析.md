---
title: PCA主成分分析
date: 2023/11/15
categories:
    - 机器学习
tags: 
    - 降维
---

PCA主成分分析是一种合理的降维算法，在减少需要分析的指标同时，尽量减少原指标包含信息的损失，以达到对所收集数据进行全面分析的目的



## PCA的概念

PCA的主要思想是将 $n$ 维特征映射到 $k$ 维上，这 $k$ 维是全新的正交特征也被称为主成分，是在原有 $n$ 维特征的基础上重新构造出来的 $k$ 维特征。PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴，新的坐标轴的选择与数据本身是密切相关的。其中，**第一个新坐标轴选择是原始数据中方差最大的方向**，**第二个新坐标轴选取是与第一个坐标轴正交的平面中使得方差最大的**，**第三个轴是与第1,2个轴正交的平面中方差最大的**。依次类推，可以得到 $n$ 个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，大部分方差都包含在前面 $k$ 个坐标轴中，后面的坐标轴所含的方差几乎为0。于是，我们可以忽略余下的坐标轴，只保留前面 $k$ 个含有绝大部分方差的坐标轴。事实上，这相当于只保留包含绝大部分方差的维度特征，而忽略包含方差几乎为 0 的特征维度，实现对数据特征的降维处理。

**思考：**我们如何得到这些包含最大差异性的主成分方向呢？

**答案：**事实上，通过计算数据矩阵的协方差矩阵，然后得到协方差矩阵的特征值特征向量，选择特征值最大(即方差最大)的k个特征所对应的特征向量组成的矩阵。这样就可以将数据矩阵转换到新的空间当中，实现数据特征的降维。

由于得到协方差矩阵的特征值特征向量有两种方法：特征值分解协方差矩阵、奇异值分解协方差矩阵，所以PCA算法有两种实现方法：基于特征值分解协方差矩阵实现PCA算法、基于SVD分解协方差矩阵实现PCA算法。



## 协方差和散度矩阵

样本均值：$\bar{x}=\frac1n\sum_{i=1}^Nx_i$

样本方差：$S^2=\frac1{n-1}\sum_{i=1}^n\left(x_i-\bar{x}\right)^2$

样本X和样本Y的协方差：
$$
\begin{aligned}
Cov\left(X,Y\right)& =E\left[\left(X-E\left(X\right)\right)\left(Y-E\left(Y\right)\right)\right]  \\
&=\frac1{n-1}\sum_{i=1}^n{(x_i-\bar{x})(y_i-\bar{y})}
\end{aligned}
$$
由上面的公式，我们可以得到以下结论：

(1) 方差的计算公式是针对一维特征，即针对同一特征不同样本的取值来进行计算得到；而协方差则必须要求至少满足二维特征；方差是协方差的特殊情况。

(2) 方差和协方差的除数是 $n-1$ ,这是为了得到方差和协方差的无偏估计。

协方差为正时，说明 $X$ 和 $Y$ 是正相关关系；协方差为负时，说明 $X$ 和 $Y$ 是负相关关系；$Cov(X,X)$ 就是 $X$ 的方差。当样本是 $n$ 维数据时，它们的协方差实际上是协方差矩阵(对称方阵)。例如，对于3维数据 $(x,y,z)$，计算它的协方差就是：
$$
Cov(X,Y,Z)=\begin{bmatrix}Cov(x,x)&Cov(x,y)&Cov(x,z)\\Cov(y,x)&Cov(y,y)&Cov(y,z)\\Cov(z,x)&Cov(z,y)&Cov(z,z)\end{bmatrix}
$$
散度矩阵定义为：
$$
S=\sum_{k=1}^{n}(\mathbf{x}_{k}-\boldsymbol{m})(\mathbf{x}_{k}-\boldsymbol{m})^{T}
$$
$$
m=\frac{1}{n}\sum_{k=1}^{n}x_{k}
$$
对于数据 $X$ 的散度矩阵为 $XX^T$ 。其实协方差矩阵和散度矩阵关系密切，散度矩阵就是协方差矩阵乘以（总数据量-1）。因此它们的**特征值**和**特征向量**是一样的。这里值得注意的是，散度矩阵是**SVD奇异值分解**的一步，因此PCA和SVD是有很大联系。


特征值分解矩阵原理和SVD分解矩阵原理参考
上一篇博客 (SVD奇异值分解)[/blogs/]

总结：
（1）求 $AA^T$ 的特征值和特征向量，用单位化的特征向量构成 $U$。
（2）求 $A^TA$ 的特征值和特征向量，用单位化的特征向量构成 $V$。
（3）将 $AA^T$ 或者 $AA^T$ 的特征值求平方根，然后构成 $Σ$。

## PCA算法两种实现方法
### 特征值分解协方差矩阵
输入：数据集 $X=\{x_1,x_2,x_3,\ldots,x_n\}$，需要降到 $k$ 维。
1) 去平均值(即去中心化)，即每一位特征减去各自的平均值。
2) 计算协方差矩阵，$\frac1nXX^T$ ，注：这里除或不除样本数量 $n$ 或 $n-1$ ，其实对求出的特征向量没有影响。
3) 用特征值分解方法求协方差矩阵 $\frac1nXX^T$ 的特征值与特征向量。
4) 对特征值从大到小排序，选择其中最大的 $k$ 个。然后将其对应的 $k$ 个特征向量分别作为行向量组成特征向量矩阵 $P$。
5) 将数据转换到 $k$ 个特征向量构建的新空间中，即 $Y=PX$。
**总结：**

1)关于这一部分为什么用 $\frac1nXX^T$ ，这里面含有很复杂的线性代数理论推导，想了解具体细节的可以看下面这篇文章。

[CodingLabs - PCA的数学原理](https://link.zhihu.com/?target=http%3A//blog.codinglabs.org/articles/pca-tutorial.html)

2)关于为什么用特征值分解矩阵，是因为 $\frac1nXX^T$ 是方阵，能很轻松的求出特征值与特征向量。当然，用奇异值分解也可以，是求特征值与特征向量的另一种方法。

举个例子
$$
X=\begin{pmatrix}-1&-1&0&2&0\\-2&0&0&1&1\end{pmatrix}
$$
以 $X$ 为例，我们用PCA方法将这两行数据降到一行。
1) 因为 $X$ 矩阵的每行已经是零均值，所以不需要去平均值。
2) 求协方差矩阵：
$$
C=\frac15\begin{pmatrix}-1&-1&0&2&0\\-2&0&0&1&1\end{pmatrix}\begin{pmatrix}-1&-2\\-1&0\\0&0\\2&1\\0&1\end{pmatrix}=\begin{pmatrix}\frac65&\frac45\\\frac45&\frac65\end{pmatrix}
$$
3) 求协方差矩阵的特征值与特征向量
	求解后的特征值为：$\lambda_1=2\text{,}\lambda_2=\frac25$
	对应的特征向量为：$c_1\begin{pmatrix}1\\1\end{pmatrix},c_2\begin{pmatrix}-1\\1\end{pmatrix}$
	其中对应的特征向量分别是一个通解， $c_1$ 和 $c_2$ 可以取任意实数。那么标准化后的特征向量为：
	$\left.\left(\begin{matrix}\frac{1}{\sqrt{2}}\\\frac{1}{\sqrt{2}}\end{matrix}\right.\right),\left(\begin{matrix}-\frac{1}{\sqrt{2}}\\\frac{1}{\sqrt{2}}\end{matrix}\right)$
4) 矩阵 $P$ 为：
$$
P =
\begin{pmatrix}
  \frac{1}{\sqrt2} & \frac{1}{\sqrt2} \\  -\frac{1}{\sqrt2} & \frac{1}{\sqrt2}
\end{pmatrix}
$$
5) 最后我们用 $P$ 的第一行乘以数据矩阵 $X$ ，就得到了降维后的表示：
$$
Y=(\frac{1}{\sqrt2}  \frac{1}{\sqrt2})\begin{pmatrix}-1&-1&0&2&0\\-2&0&0&1&1\end{pmatrix}=\begin{pmatrix} -\frac{3}{\sqrt{2}}&-\frac{1}{\sqrt{2}}&0&\frac{3}{\sqrt{2}}&-\frac{1}{\sqrt{2}}\end{pmatrix}
$$
注意：如果我们通过特征值分解协方差矩阵，那么我们只能得到一个方向的PCA降维。这个方向就是对数据矩阵 $X$ 从行(或列)方向上压缩降维。

### SVD分解协方差矩阵
输入：数据集 $X=\{x_1,x_2,x_3,\ldots,x_n\}$，需要降到 $k$ 维。
1) 去平均值，即每一位特征减去各自的平均值。
2) 计算协方差矩阵。
3) 通过SVD计算协方差矩阵的特征值与特征向量。
4) 对特征值从大到小排序，选择其中最大的 $k$ 个。然后将其对应的 $k$ 个特征向量分别作为列向量组成特征向量矩阵。
5) 将数据转换到k个特征向量构建的新空间中。
在PCA降维中，我们需要找到样本协方差矩阵 $XX^T$ 的最大 $k$ 个特征向量，然后用这最大的 $k$ 个特征向量组成的矩阵来做低维投影降维。可以看出，在这个过程中需要先求出协方差矩阵 ,当样本数多、样本特征数也多的时候，这个计算还是很大的。当我们用到SVD分解协方差矩阵的时候，SVD有两个好处：
1. 有一些SVD的实现算法可以先不求出协方差矩阵 $XX^T$ 也能求出我们的右奇异矩阵 $V$。也就是说，我们的PCA算法可以不用做特征分解而是通过SVD来完成，这个方法在样本量很大的时候很有效。实际上，scikit-learn的PCA算法的背后真正的实现就是用的SVD，而不是特征值分解。
2. 注意到PCA仅仅使用了我们SVD的左奇异矩阵，没有使用到右奇异值矩阵，那么右奇异值矩阵有什么用呢？
假设我们的样本是 $m\times n$ 的矩阵X，如果我们通过SVD找到了矩阵 $X^TX$ 最大的 $k$ 个特征向量组成的 $k\times n$ 的矩阵 $V^T$,则我们可以做如下处理：
$$
X_{m\times k}^{^{\prime}}=X_{m\times n}V_{n\times k}^T
$$
可以得到一个 $m\times k$ 的矩阵 $X^{\prime}$ ,这个矩阵和我们原来 $m\times k$ 的矩阵 $X$ 相比，列数从 $n$ 减到了 $k$ ，可见对列数进行了压缩。也就是说，<mark>左奇异矩阵可以用于对行数的压缩；右奇异矩阵可以用于对列(即特征维度)的压缩</mark>。这就是我们用SVD分解协方差矩阵实现PCA可以得到两个方向的PCA降维(即行和列两个方向)。

### PCA示例
```python
##Python实现PCA
import numpy as np
def pca(X, k):#k is the components you want
  # mean of each feature
  n_samples, n_features = X.shape
  mean = np.array([np.mean(X[:,i]) for i in range(n_features)])
  # normalization
  norm_X = X-mean
  # scatter matrix
  scatter_matrix = np.dot(np.transpose(norm_X),norm_X)
  # Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature = np.array([ele[1] for ele in eig_pairs[:k]])
  # get new data
  data = np.dot(norm_X,np.transpose(feature))
  return data

X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

print(pca(X,1))
```

```python
# 用sklearn的PCA
from sklearn.decomposition import PCA
import numpy as np
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=1)
pca.fit(X)
print(pca.transform(X))
```

结果并不一样
sklearn中的PCA是通过svd_flip函数实现的，sklearn对奇异值分解结果进行了一个处理，因为$u_i\times σ_i \times v_i=(-u_i)\times σ_i\times (-v_i)$，也就是 $u$ 和 $v$ 同时取反得到的结果是一样的，而这会导致通过PCA降维得到不一样的结果（虽然都是正确的）。

## PCA的理论推导
PCA有两种通俗易懂的解释：(1)最大方差理论；(2)最小化降维造成的损失。这两个思路都能推导出同样的结果。

最大方差理论：
在信号处理中认为信号具有较大的方差，噪声有较小的方差，信噪比就是信号与噪声的方差比，越大越好。样本在 $u_1$ 上的投影方差较大，在 $u_2$ 上的投影方差较小，那么可认为 $u_2$ 上的投影是由噪声引起的。

因此我们认为，最好的 $k$ 维特征是将 $n$ 维样本点转换为 $k$ 维后，每一维上的样本方差都很大。

比如我们将下图中的5个点投影到某一维上，这里用一条过原点的直线表示（数据已经中心化）：
![图片.png](https://s2.loli.net/2023/11/17/lLCRqI1fhj4ydVG.png)
假设我们选择两条不同的直线做投影，那么左右两条中哪个好呢？根据我们之前的方差最大化理论，左边的好，因为投影后的样本点之间方差最大（也可以说是投影的绝对值之和最大）。
计算投影的方法见下图：
![图片.png](https://s2.loli.net/2023/11/17/ncTNrEIqsX6V57K.png)
图中，红色点表示样例，蓝色点表示在 $u$ 上的投影，$u$ 是直线的斜率也是直线的方向向量，而且是单位向量。蓝色点是在 $u$ 上的投影点，离原点的距离是<x,u>（即 $X^TU$ 或者 $U^TX$ ）。

## 选择降维后的维度K
如何选择主成分个数K呢？先来定义两个概念：
$$
MSE = \frac{1}{m}\sum_{i=1}^{m}\left\|x^{(i)}-x_{approx}^{(i)}\right\|^{2}
$$
$$
\text{总体误差} =\frac{1}{m}\sum_{i=1}^{m}\left\|x^{(i)}\right\|^{2}
$$
选择不同的 $K$ 值，然后用下面的式子不断计算，选取能够满足下列式子条件的最小 $K$ 值即可。
$$
\frac{\frac{1}{m}\sum_{i=1}^{m}\left\|x^{(i)}-x_{approx}^{(i)}\right\|^{2}}{\frac{1}{m}\sum_{i=1}^{m}\left\|x^{(i)}\right\|^{2}}\leq t
$$其中 $t$ 值可以由自己定，比如 $t$ 值取0.01，则代表了该PCA算法保留了99%的主要信息。当你觉得误差需要更小，你可以把 $t$ 值设置的更小。上式还可以用SVD分解时产生的 $S$ 矩阵来表示，如下面的式子：
$$
1-\frac{\sum_{i=1}^{k}S_{ii}}{\sum_{i=1}^{n}S_{ii}}\leq t
$$