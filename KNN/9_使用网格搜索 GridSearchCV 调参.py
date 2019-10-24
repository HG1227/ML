#!/usr/bin/python
#coding:utf-8

"""
@software: PyCharm
@file: 9_使用网格搜索 GridSearchCV 调参.py
"""


'''
好在，在 Sklearn 中有一个网格搜索函数（GridSearchCV）能够很轻松解决这个问题。
我们只需要添加超参数即可，它会自动匹配超参数之间的关系，让模型正确运行，
最终返回最好的一组超参数组合。
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
KNN_clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')
KNN_clf.fit(X_train, y_train)
score = KNN_clf.score(X_test, y_test)
print(score)


#定义超参数
#第一步，在 GridSearchCV 函数的 param_grid 参数中定义超参数。
# 该参数是一个数组，数组中包括一个个字典，
# 我们只需要把每一组超参数组合放置在该字典中即可。
# 字典的键为超参数名称，键的值为超参数的搜索取值范围。

param_frid=[
    {
        'n_neighbors':[i for i in range(1,21)],
        'weights':['uniform']
    },

    {
        'n_neighbors': [i for i in range(1, 21)],
        'weights': ['distance'],
        'p':[p for p in range(1,11)]
    }

]

'''
比如，第一个字典中有两个超参数：n_neighbors 和 weights 。
n_neighbors 取值范围是 1-20，weights 键值 uniform 表示不考虑距离，
这组参数会循环搜索 20 次。
第二个字典中 weights 变为 distance 表示要考虑距离，
n_neighbors 和 p 两个超参数循环 20*10 共 200 次。
两个字典加起来一共要循环搜索 220 次，后续就会建立 220 个模型中
并根据得分返回最好的超参数组合。
'''

#建模
'''
第二步，建模。先创建默认的 kNN 模型，
把模型和刚才的超参数传进网格搜索函数中，
接着传入训练数据集进行 fit 拟合，这时模型会尝试 
220 组超参数组合，数据大的话会比较费时，

'''


# estimator = LGBMRegressor(
#         num_leaves = 50, # cv调节50是最优值
#         max_depth = 13,
#         learning_rate =0.1,
#         n_estimators = 1000,
#         objective = 'regression',
#         min_child_weight = 1,
#         subsample = 0.8,
#         colsample_bytree=0.8,
#         nthread = 7,
#     )








from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(score,param_grid=param_frid)

grid_search.fit(X_train,y_train)


#模型训练好之后可以通过它的方法查看训练结果。
bs=grid_search.best_score_
print(bs)
bp=grid_search.best_params_
be=grid_search.best_estimator_

#根据网络搜索的最佳超参数建模
KNN_clf=grid_search.best_estimator_
KNN_clf.score(X_test,y_test)

#更多网格搜索参数可以在官网了解它的 API，之后我们在讲交叉验证（Cross-Validation，GridSearchCV 的 CV）的时候还会再来介绍它。
#best_params_参数查看最佳超参数组合：
#'n_neighbors': 4, 'p': 1, 'weights': 'distance'