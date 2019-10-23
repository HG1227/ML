#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 020.py
"""
# !/usr/bin/python
# coding:utf-8


'''
首先从 sklearn 中引入了 kNN 的分类算法函数 KNeighborsClassifier
并建立模型，设置最近的 K 个样本数量 n_neighbors 为 3。
接下来 fit 训练模型，最后 predict 预测模型得到分类结果
'''

import numpy as  np
from sklearn import datasets

# 加载葡萄酒数据集
win = datasets.load_wine()

# 查看wine能够调用的方法
print(win.keys())
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

X = win.data
print(X.shape)  #  (178, 13)

y = win.target
print(y.shape)  # (178,)

# 先来调用 Sklearn 数据集划分函数 train_test_split ：
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=321)
# test_size 是测试集比例，random_state 是一个随机种子，保证每次运行都能得到一样的随机结果


print(X_train.shape)  # (124, 13)
print(X_test.shape)  # (54, 13)
print(y_train.shape)  # (124,)
print(y_test.shape)  # (54,)

from sklearn.neighbors import KNeighborsClassifier

KNN_clf = KNeighborsClassifier(n_neighbors=3)
KNN_clf.fit(X_train, y_train)
y_predict = KNN_clf.predict(X_test)





from sklearn.metrics import accuracy_score

sc = accuracy_score(y_test, y_predict)
print(sc)  # 0.7592592592592593

# 当不关心y_predict 时，可以用另一种简单的方法计算score
sc = KNN_clf.score(X_test, y_test)
print(sc)  # 0.7592592592592593
'''
accuracy_score 函数利用 y_test 和 y_predict 计算出得分，
这种方法需要先计算出 y_predict。而有些时候我们并不需要知道 y_predict ，
而只关心模型最终的得分，那么可以跳过 predict 直接计算得分么，
答案是可以的，就是第二种方法。
一行代码就能计算出来，更为简洁：
kNN_clf.score(X_test,y_test)
'''

