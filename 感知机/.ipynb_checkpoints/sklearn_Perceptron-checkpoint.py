#!/usr/bin/python
#coding:utf-8
#@software: PyCharm
#@file: sklearn_Perceptron.py
#@time: 2019/12/4


from sklearn.datasets import make_classification

x,y = make_classification(n_samples=1000, n_features=2,n_redundant=0,
                          n_informative=1,n_clusters_per_class=1)
 
# 使用sklearn中的make_classification来生成一些用来分类的样本