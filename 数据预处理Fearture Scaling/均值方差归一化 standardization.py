#!/usr/bin/python
#coding:utf-8

"""
@software: PyCharm
@file: 均值方差归一化 standardization.py
"""


'''
这种方法是把所有数据映射到 均值为 0 方差为 1 的分布中。
即使有极端值存在，也不会出现严重的数据有偏分布，便于后续建模。

Xscl=(X-Xmean)/σ

σ:标准差(又名均方差)
⒈方差s^2=[（x1-x）^2+（x2-x）^2+......（xn-x）^2]/（n）（x为平均数）
⒉标准差=方差的算术平方根errorbar。

'''

import  numpy as np
from  math import sqrt
x=np.array([1,1,2,2,3,3,100]).reshape(-1,1)
x_s=sqrt(sum((x-x.mean())**2)/len(x))
x=(x-x.mean())/x_s

# print(x)


from  sklearn import  datasets
from  sklearn.model_selection import train_test_split

wine=datasets.load_wine()
# print(wine.DESCR)
X=wine.data
y=wine.target
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               random_state=321)

from  sklearn.neighbors import  KNeighborsClassifier

KNN_clf=KNeighborsClassifier(n_neighbors=3)
np.set_printoptions(suppress=True)

# Sklearn 均值方差归一化
from sklearn.preprocessing import StandardScaler
standarscaler=StandardScaler()
standarscaler.fit(X_train)

#fit之后查看相关值得情况   均值，方差等
sm=standarscaler.mean_
print(sm)


#标准差
ss=standarscaler.scale_
print(ss)


X_train_standar=standarscaler.transform(X_train)
KNN_clf.fit(X_train_standar,y_train)

#注意两点：
#1 测试前也必须对测试集归一化
#2 归一化采用的训练集的均值和方差归一化测试集，而不是测试集的均值和方差

X_test_standar=standarscaler.transform(X_test)

sc=KNN_clf.score(X_test_standar,y_test)
print(sc)       #0.9629629629629629

y_predict=KNN_clf.predict(X_test_standar)
#测试集中分类正确的样本数
print('样本数：',len(y_test))
ss=sum(y_predict==y_test)
print('正确数：',ss)
