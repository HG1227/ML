#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 多元线性回归LinearRegression.py
"""

# 1. 导入需要的模块和库

import pandas as pd
from sklearn.datasets import fetch_california_housing as fch  # 加利福尼亚房屋价值数据集
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split

# 2. 导入数据，探索数据
housevalue = fch()  # 会需要下载，大家可以提前运行试试看
X = pd.DataFrame(housevalue.data)  # 放入DataFrame中便于查看
y = housevalue.target
# print(X.shape)

X.columns = housevalue.feature_names

# 3. 分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# 重新建立索引
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

# 如果希望进行数据标准化，还记得应该怎么做吗？
# 先用训练集训练标准化的类，然后用训练好的类分别转化训练集和测试集


# 4. 建模
reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)

# 5. 探索建好的模型
# reg.coef_ 参数w1,w2.....wn
print(reg.coef_)

# 截距  reg.intercept_
print(reg.intercept_)

from sklearn.metrics import mean_squared_error as MSE

mse = MSE(yhat, Ytest)
print(mse)

from sklearn.model_selection import cross_val_score

# 当scoring='mean_squared_error'会报错了
cv_mse = cross_val_score(reg, X, y, scoring='neg_mean_squared_error')
print(cv_mse)

from sklearn.metrics import r2_score

# 两个的结果不一样
m_r2 = r2_score(yhat, Ytest)
print(m_r2)  # 0.33806537615559895

r2 = reg.score(Xtest, Ytest)
print(r2)  # 0.6043668160178816

# 正确的调用方式
r2 = r2_score(Ytest, yhat)
# 或者
# r2=r2_score(y_true=Ytest,y_pred=yhat)
print(r2)  # 0.6043668160178816

r2 = cross_val_score(reg, X, y, cv=10, scoring='r2').mean()
print(r2)  # 0.5110068610524554

import matplotlib.pyplot as plt

sorted(Ytest)
plt.plot(range(len(Ytest)), sorted(Ytest), c="black", label="Data")
plt.plot(range(len(yhat)), sorted(yhat), c="red", label="Predict")
plt.legend()
plt.show()

import numpy as np
rng = np.random.RandomState(42)
X = rng.randn(100, 80)
y = rng.randn(100)
r2=cross_val_score(LR(), X, y, cv=5, scoring='r2')
print(r2)
#[-179.86577271   -5.69860535  -15.10281588  -78.21750079  -70.19186257]
