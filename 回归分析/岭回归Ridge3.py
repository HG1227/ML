#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 岭回归Ridge3.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing as fch
from sklearn.linear_model import RidgeCV

housevalue = fch()
X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目"
    , "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]
Ridge_ = RidgeCV(alphas=np.arange(1, 1001, 100)
                 # ,scoring="neg_mean_squared_error"
                 , store_cv_values=True
                 # ,cv=5
                 ).fit(X, y)

# 无关交叉验证的岭回归结果
re = Ridge_.score(X, y)
print(re )
# 调用所有交叉验证的结果
re = Ridge_.cv_values_.shape
print(re )
# 进行平均后可以查看每个正则化系数取值下的交叉验证结果
re = Ridge_.cv_values_.mean(axis=0)
print(re )
# 查看被选择出来的最佳正则化系数
re = Ridge_.alpha_
print(re )

'''
0.6060251767338429
(20640, 10)
[0.52823795 0.52787439 0.52807763 0.52855759 0.52917958 0.52987689
 0.53061486 0.53137481 0.53214638 0.53292369]
101
'''
