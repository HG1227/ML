# -*- encoding: utf-8 -*-
"""
@software: PyCharm
@file : svm.py
@time : 2019/12/31 
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt


class SVM(object):
    def __init__(self, max_iter=100, kernel='linear'):
        self.max_iter = max_iter
        self._kernel = kernel

    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = .0

        # 将Ei 保存到一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]

        # 松弛变量
        self.C = 1.0

    def _KKT(self,i):
        y_g = self._g(i) *self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g  == 1
        else:
            return y_g  <= 1

    def _g(self,i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j]*self.Y[j]*self.kernel(self.X[i],self.X[j])
        return r

    # 核函数
    def kernel(self, x1,x2):
        if self._kernel == 'linear':
            return sum([x1[k]*x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k]*x2[k] for k in range(self.n)])+1)**2

        return 0

    def _E(self,i):
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0<self.alpha[i] <self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self._KKT(i):
                continue
            E1 = self.E(i)
            # 如果E2是+，选择最小的；如果E2是负的，选择最大的


    def _compare(self,_alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self,features, labels):
        self.init_args(features,labels)

        for i in range(self.max_iter)
            # train
            i1, i2 = self._init_alpha()










if __name__ == '__main__':
    pass