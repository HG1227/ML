#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 7_模型参数_weights.py
"""

'''
所以，在建立模型时，还可以从距离出发，将样本权重与距离挂钩，
近的点权重大，远的点权重小。
怎么考虑权重呢，可以用取倒数这个简单方法实现。
假设一个点
绿点距离是 1，取倒数权重还为 1；两个红点的距离分别是 3 和 4，
去倒数相加后红点权重和为 7/12。绿点的权重大于红点，所以黄点属于绿色类。
'''

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# 加载葡萄酒数据集
win = datasets.load_wine()

# 查看wine能够调用的方法
# print(win.keys())
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])

X = win.data
# print(X.shape)  # (178, 13)

y = win.target
# print(y.shape)  # (178,)

# 先来调用 Sklearn 数据集划分函数 train_test_split ：
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=321)
# test_size 是测试集比例，random_state 是一个随机种子，保证每次运行都能得到一样的随机结果


'''
在 Sklearn 的 kNN 模型中有专门考虑权重的超参数：weights，它有两个选项：
    * uniform 不考虑距离权重，默认值
    * distance 考虑距离权重
'''

# 从这两个维度，我们可以再次循环遍历找到最合适的超参数：
best_k = 0
best_score = 0
best_method = None
for method in [ 'uniform', 'distance']:
    for k in range(1, 10):
        KNN_clf = KNeighborsClassifier(k, weights=method)
        KNN_clf.fit(X_train, y_train)
        score = KNN_clf.score(X_test, y_test)
        if score > best_score:
            best_method = method
            best_score = score
            best_k = k

print('best_method:', best_method)
print('best_k:', best_k)
print('best_score:', best_score)
