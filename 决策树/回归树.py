#!/usr/bin/python
# coding:utf-8

"""
@software:  PyCharm
@file: 回归树.py
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=321)

# 交叉验证cross_val_score的用法
cross_val_score(regressor, boston.data, boston.target
                , cv=10
                , scoring='neg_mean_squared_error')
