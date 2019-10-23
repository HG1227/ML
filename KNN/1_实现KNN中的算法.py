#!/usr/bin/python
#coding:utf-8

"""
@software: PyCharm
@file: 实现KNN中的算法.py
"""

import numpy as np
X_raw = [[14.23, 5.64],
         [13.2, 4.38],
         [13.16, 5.68],
         [14.37, 4.80],
         [13.24, 4.32],
         [12.07, 2.76],
         [12.43, 3.94],
         [11.79, 3.],
         [12.37, 2.12],
         [12.04, 2.6]]

y_raw = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

x_test = np.array([12.8, 4.1])

X_train = np.array(X_raw)
y_train = np.array(y_raw)

import matplotlib.pyplot as plt

plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], s=100, c='r'
            , label="A")

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], s=100, c='g'
            , label="B")

plt.scatter(x_test[0], x_test[1], s=100, c='y', label="test")
plt.legend()
plt.show()

 
# 根据欧拉公式计算黄色的新样本点到每个样本点的距离：
from math import sqrt

distance = [sqrt(np.sum((x - x_test) ** 2)) for x in X_train]

# 接着找出最近的 3 个点，可以使用 np.argsort 函数返回样本点的索引位置：
# numpy.argsort() 函数返回的是数组值从小到大的索引值。
sort = np.argsort(distance)
print(sort)

# 通过这个索引值就能在 y_train 中找到对类别，再统计出排名前 3 的就行了：
# 得到最近三个的类别
k = 3
topK = [y_train[i] for i in sort[:k]]

# 使用 Counter 函数统计返回类别值即可：
from collections import Counter

votes = Counter(topK)
print(votes)  # Counter({0: 2, 1: 1})
predict_y = votes.most_common(1)[0][0]
print(votes.most_common(1))  # [(0, 2)]
print(predict_y)  # 0   也就是类别



# 将上面的算法 改写为函数
def KNNClassify(K, X_train, y_train, X_predict):
    distance = [sqrt(np.sum((x - x_test) ** 2)) for x in X_train]
    sort = np.argsort(distance)
    votes = Counter(topK)
    predict_y = votes.most_common(1)[0][0]
    return predict_y


# 将上面的算法 改写为函数
class kNNClassifier:
    def __init__(self, k):
        '''
        :param k: k 表示我们要选择传进了的 k 个近邻点
        '''
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        '''
        接着定义一个 fit 函数，这个函数就 是用来拟合 kNN 模型，
        但 kNN 模型并不需要拟合，所以我们就原封不动地把数据集复制一遍，
        最后返回两个数据集自身。
        :param X_train:
        :param y_train:
        :return:
        '''

        # "添加 assert 断言是为了确保输入正常的数据集和k值，
        # 如果不添加一旦输入不正常的值，难找到出错原因"
        assert X_train.shape[0] == y_train.shape[0]

        ## 只输出行数shape[0]
        assert self.k <= X_train.shape[0]

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict1(self, X_predict):
        #"要求predict 之前要先运行 fit 这样self._X_train 就不会为空"
        assert self._X_train is not None
        assert self._y_train is not None

        #"要求测试集和预测集的特征数量一致"
        assert X_predict.shape[1]==self._X_train.shape[1]

        distance = [sqrt(np.sum((x_train - X_predict) ** 2)) for  x_train in self._X_train]
        sort = np.argsort(distance)
        topK = [self._y_train[i] for i in sort[:self.k]]
        votes = Counter(topK)
        predict_y = votes.most_common(1)[0][0]
        return predict_y

    #再进一步，如果我们一次预测不只一个点，而是多个点，属于哪一类：
    #改写上述函数
    def predict2(self,X_predict):
        y_predict=[self._predict(x) for x in X_predict]
        return y_predict

    def _predict(self,x):
        assert self._X_train is not None
        assert self._y_train is not None

        distance = [sqrt(np.sum((x_train - x) ** 2)) for  x_train in self._X_train]
        sort = np.argsort(distance)
        topK = [self._y_train[i] for i in sort[:self.k]]
        votes = Counter(topK)
        predict_y = votes.most_common(1)[0][0]
        print(predict_y)
        return predict_y











