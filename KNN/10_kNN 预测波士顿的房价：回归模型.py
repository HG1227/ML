#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 10_kNN 预测波士顿的房价：回归模型.py
"""

# 以波士顿房价数据集为例，使用 kNN 模型解决回归问题——预测房价。


# 在 sklearn 中使用 KNeighborsClassifier 类解决分类，
# 回归问题则可以调用 KNeighborsRegressor类。

# 计算思路也很简单，主要分两步。
# 第一步和分类算法一样，找到离待预测节点最近的 K 个点，
# 第二步则是取这 K 个节点值的平均值作为待预测点的预测值。


'''
波士顿房价是机器学习中很常用的一个解决回归问题的数据集。
数据统计于 1978 年，包括 506个房价样本，每个样本包括波士顿不同郊区房屋的
13 种特征信息，比如：住宅房间数、城镇教师和学生比例等。标签值则是每栋房子的房价
（千美元）。所以这是一个小型数据集，有 506 * 14 维。
我们通过这几步来预测房价：
	* 加载数据集并初步探索
	* 划分训练集和测试集
	* 对特征做均值方差归一化
	* 建立 kNN 回归模型并预测

'''

'''
样本一共有 13 个特征，建立模型时可以纳入全部特征也可以只纳入部分，我们选择后者。使用 SelectKBest 方法可以筛选出和标签最相关的 K 个特征，这里选择和房价最相关的 3 个特征：
	* RM
	* PTRATIO
	* LSTAT
'''

from sklearn.feature_selection import SelectKBest, f_regression
import  numpy as np
selector = SelectKBest(f_regression, k=3)
X = []
y = []
X_new = selector.fit(X, y)

# 返回最相关的三个特征的索引
best = selector.get_support(indices=True).tolist()

#特征选择好之后，接下来划分数据集并归一化，然后建模，代码如下：

#划分训练集
from  sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,
                                               y,
                                               test_size=0.3,
                                               random_state=321)

#均值方差归一化
from  sklearn.preprocessing import  StandardScaler
standarsclar=StandardScaler()
standarsclar.fit(X_train)
X_train_std=standarsclar.transform(X_train)
X_test_std=standarsclar.transform(X_test)

#建立KNN模型
from  sklearn.neighbors import KNeighborsRegressor

#使用最简单的默认超参数建模
KNN_clf=KNeighborsRegressor()
KNN_clf.fit(X_train_std,y_train)

y_pred=KNN_clf.predict(X_test_std)

from sklearn.metrics import mean_squared_error
#利用均方根误差 判断模型效果
s=np.sqrt(mean_squared_error(y_test,y_pred))



#回归模型可用R2值 评估模型的拟合程度效果
#R2越接近1  表示效果越好

from  sklearn.metrics import  r2_score

r2=r2_score(y_test,y_pred)

