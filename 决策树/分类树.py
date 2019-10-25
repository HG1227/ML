#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 分类树.py
"""


#1. 导入需要的算法库和模块
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import  pandas as pd


#2. 探索数据
wine = datasets.load_wine()
#如果wine是一张表，应该长这样：
# tb=pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
# print(tb)
# wine.feature_names
# wine.target_names


#3. 分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)

#4. 建立模型
clf = tree.DecisionTreeClassifier(criterion='entropy'
                                  ,random_state=321
                                  ,splitter='random'
                                  ,min_samples_leaf=10
                                  ,min_samples_split=10)
clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
print(score)        #0.9074074074074074

print(clf.apply(Xtest))

#5. 画出一棵树吧
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类'
    ,'花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']

with open('tee.dot', 'w') as dot_tree:
    dot_tree=tree.export_graphviz(clf,
                                  feature_names=feature_name,
                                  class_names=['胡红光','胡希奥','话还是'],
                                  filled=True,
                                  rounded=True,
                                  out_file=dot_tree,
                                  special_characters=True

    )


#6. 探索决策树
#特征重要性
# clf.feature_importances_

fi=[*zip(feature_name,clf.feature_importances_)]
# for i in fi:
#     print(i)
