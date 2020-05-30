## 机器学习-感知机`perceptron`

在机器学习中，**感知机**（perceptron）是二分类的线性分类模型，属于**监督学习算法**。

输入为实例的特征向量，输出为实例的类别（取+1和-1）。感知机对应于输入空间中将实例划分为两类的分离超平面。感知机旨在求出该超平面，为求得超平面导入了基于误分类的损失函数，利用梯度下降法 对损失函数进行最优化（最优化）。感知机的学习算法具有简单而易于实现的优点，分为原始形式和对偶形式。感知机预测是用学习得到的感知机模型对新的实例进行预测的，因此属于判别模型。

感知机由**Rosenblatt**于1957年提出的，是**神经网络**和**支持向量机**的基础。



## 定义

假设输入空间(特征向量)为X⊆$R^n$，输出空间为Y={-1, +1}。输入x∈ X表示实例的特征向量，对应于输入空间的点；输出y∈Y表示示例的类别。由输入空间到输出空间的函数为:
$$
f(x) = sign(w·x + b)
$$
称为感知机, 参数 $w$ 叫做权值向量 **weight**，$b$ 称为偏置 **bias**, $w·x$表示w和x的**点积**

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204192113.png)

sign为符号函数，即

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204192043.png)

在二分类问题中, $f(x)$的值（+1或-1）用于分类 $x$ 为正样本（+1）还是负样本（-1）。**感知机是一种线性分类模型**，属于判别模型。

我们需要做的就是找到一个最佳的满足$w·x + b =0$ 的 w 和 b 值，即**分离超平面（*separating hyperplane*）**。如下图，一个线性可分的感知机模型

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204190927.png)

中间的直线即 $w·x + b =0$

线性分类器的几何表示有：直线、平面、超平面。



## **学习策略**

> **核心：极小化损失函数。**

如果训练集是可分的，感知机的学习目的是求得一个能将训练集正实例点和负实例点完全分开的分离超平面。为了找到这样一个平面（或超平面），即确定感知机模型参数 $w$  和 $b$，我们采用的是损失函数，同时并将损失函数极小化。

对于损失函数的选择，采用的是误分类点到超平面的距离（，这里采用的是几何间距，就是点到直线的距离）：

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204191846.png)

其中||$w$||是 $L_2$ 范数

对于误分类点 $(x_i,y_i)$ 来说：

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204192444.png)

成立，因为当 $w·x +b >0 $ 时，$y_i$= -1，当$w·x +b <0 $ 时，$y_i$ = +1, 因此误分类点到超平面的距离为：

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204193007.png)

那么，所有点到超平面的总距离为：

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204193048.png)

不考虑 $1/||w|| $  就得到感知机的损失函数了。

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204193248.png)



其中M为误分类的集合。这个损失函数就是感知机学习的**经验风险函数**。

可以看出，随时函数 $L(w,b)$ 是非负的。**如果没有误分类点，则损失函数的值为0，而且误分类点越少，误分类点距离超平面就越近，损失函数值就越小**。同时，损失函数 $L(w,b)$ 是连续可导函数。

## 学习算法

感知机学习转变成求解损失函数 $L(w,b)$ 的最优化问题。最优化的方法是**随机梯度下降法（stochastic gradient descent）**，这里采用的就是该方法。

感知机学习算法本身是误分类驱动的，因此我们采用随机梯度下降法。首先，任选一个超平面$w_0$和$b_0$，然后使用梯度下降法不断地**极小化目标函数**

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204193720.png)

极小化过程不是一次使M中所有误分类点的梯度下降，而是一次随机的选取一个误分类点使其梯度下降。使用的规则为为$ θ:=θ−α∇_ℓ(θ)$, 其中α是步长，$∇_θℓ(θ)$是梯度。假设误分类点集合M是固定的，那么损失函数 $L(w,b)$ 的梯度通过偏导计算：

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204194010.png)

然后，随机选取一个误分类点，根据上面的规则，计算新的 $w,b$，然后进行更新：

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204194110.png)

其中 $η$ 是步长，大于0小于1，在统计学习中称之为**学习率（*learning rate*）**。这样，通过迭代可以期待损失函数 $L(w,b)$ 不断减小，直至为0.

## **算法：感知机学习算法原始形式**

```
输入：T={(x1,y1),(x2,y2)...(xN,yN)}（其中xi∈X=Rn，yi∈Y={-1, +1}，i=1,2...N，学习速率为η）
输出：w, b;感知机模型f(x)=sign(w·x+b)
(1) 初始化w0,b0，权值可以初始化为0或一个很小的随机数
(2) 在训练数据集中选取（x_i, y_i）
(3) 如果yi(w·xi+b)≤0
           w = w + η*y_i*x_i
           b = b + η*y_i
(4) 转至（2）,直至训练集中没有误分类点

```

## 核心算法——随机梯度下降

```python
# 随机梯度下降算法
numLines = dataSet.shape[0]
numFearures = dataSet.shape[1]

w = np.zeros((1, numFearures - 1))  # initialize weights
separated = False

i = 0
while not separated and i < numLines:
    if dataSet[i][-1] * np.sum(w * dataSet[i, 0:-1]) <= 0:  # 如果分类错误
        w = w + dataSet[i][-1] * dataSet[i, 0:-1]  # 更新权重向量
        separated = False  # 设置为未完全分开
        i = 0  # 重新开始遍历每个数据点
        else:
            i += 1  # 如果分类正确，检查下一个数据点
```

## **sklearn.linear_model.Perceptron**

```python
class sklearn.linear_model.Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, 
                                      max_iter=1000, tol=0.001, shuffle=True,
                                      verbose=0, eta0=1.0, n_jobs=None, 
                                      random_state=0, early_stopping=False, 
                                      validation_fraction=0.1, n_iter_no_change=5, 
                                      class_weight=None, warm_start=False)
```

**主要参数**

- **`fit_intercept`** ：布尔

  是否应该估计拦截。 如果为假，则假定数据已居中。 默认为True。

- **`max_iter`** ：int，可选

  训练数据（又称时代）的最大次数。 它只会影响`fit`方法的行为，而不会影响`partial_fit` 。 默认为5.默认值为0.21，或者如果tol不是None。

- **`shuffle`** ：bool，可选，默认为True

  训练数据是否应该在每个循环后洗牌。

**属性**

- `.coef_`  查看训练结果，权重矩阵
- `.intercept_`  #超平面的截距，