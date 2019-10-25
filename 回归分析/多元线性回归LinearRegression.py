#!/usr/bin/python
#coding:utf-8

"""
@software: PyCharm
@file: 多元线性回归LinearRegression.py
"""

#1. 导入需要的模块和库

from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch #加利福尼亚房屋价值数据集
import pandas as pd


 #2. 导入数据，探索数据
housevalue = fch() #会需要下载，大家可以提前运行试试看
X = pd.DataFrame(housevalue.data) #放入DataFrame中便于查看
y = housevalue.target
print(X.shape)

X.columns = housevalue.feature_names


#3. 分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=420)

#重新建立索引
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

#如果希望进行数据标准化，还记得应该怎么做吗？
#先用训练集训练标准化的类，然后用训练好的类分别转化训练集和测试集


#4. 建模
reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)


#5. 探索建好的模型
#reg.coef_ 参数w1,w2.....wn
print(reg.coef_)

#截距  reg.intercept_
print(reg.intercept_)


