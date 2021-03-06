#!/usr/bin/python
#coding:utf-8



"""
手写 train_test_split 函数

观察到原始数据集的标签值是从 0 到 2 有序排列的，所以不能直接划分，
要先把数据集打乱保证随机抽样。打 乱可以用 numpy 的 permutation 函数，
它会返回打乱后的数据集的索引，
这个函数的妙处就在于根据索引就能同时匹配到 X 和 y。二者是一一对应的。
"""

import numpy as  np
from sklearn import datasets

# 加载葡萄酒数据集
win = datasets.load_wine()
X = win.data
y = win.target
np.random.seed(321)
shuffle_index = np.random.permutation(len(X))
print(shuffle_index)

from math import ceil

test_ratio = 0.3  # 测试集比例
test_size = ceil(len(X) * test_ratio)  # int向下取整，ceil向上取整

# 由于y值0-2有序排列，数据集不是随机的，故在划分前需要打乱
# 可以用shuffle函数

test_index = shuffle_index[:test_size]
train_index = shuffle_index[test_size:]

X_test = X[test_index]
print(X_test.shape)  # (54, 13)

X_train = X[train_index]
print(X_train.shape)  # (124, 13)

y_train = X[train_index]
y_test = X[test_index]
'''
因 Sklearn 中的 train_test_split 是向上取整，所以为了保持一致使用了向上取整的 ceil 函数，
ceil (3.4) = 4；int 虽然也可以取整但它是向下取整的， int (3.4) = 3。
在打乱数据之前，添加了一行 random.seed (321) 函数，
它的作用是保证重新运行时能得到相同的随机数，
若不加这句代码，每次得到的结果都会不一样，不便于相互比较。
'''

'''
封装 train_test_split 函数
'''

import numpy as  np
from math import ceil


def train_test_split(X, y, test_raton=0.3, seed=None):
    assert X.shape[0] == y.shape[0], 'X y 的行数要一样'

    if seed:
        np.random.seed(seed)

    shuffle_index = np.random.permutation(len(X))

    print(shuffle_index)
    test_size = ceil(len(X) * test_ratio)
    test_index = shuffle_index[:test_size]
    train_index = shuffle_index[test_size:]

    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]
