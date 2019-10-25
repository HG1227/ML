#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 泰坦尼克号幸存者的预测.py
"""

# 1. 导入所需要的库
# 1. 导入所需要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# 2. 导入数据集，探索数据
data = pd.read_csv('data.csv', index_col=0)
print(data.info())

# 3. 对数据集进行预处理
# 删除缺失值过多的列，和观察判断来说和预测的y没有关系的列
data.drop(['Cabin', 'Name', 'Ticket'], inplace=True, axis=1)
# print(data.info())

##处理缺失值，对缺失值较多的列进行填补，有一些特征只确实一两个值，可以采取直接删除记录的方法
data['Age'] = data["Age"].fillna(data["Age"].mean())
data = data.dropna()

# 将分类变量转换为数值型变量

# 将二分类变量转换为数值型变量

# astype能够将一个pandas对象转换为某种类型，和apply(int(x))不同，astype可以将文本类转换为数字，用这
# 个方式可以很便捷地将二分类特征转换为0~1
data["Sex"] = (data["Sex"] == "male").astype("int")

# 将三分类变量转换为数值型变量
labels = data["Embarked"].unique().tolist()
data["Embarked"] = data["Embarked"].apply(lambda x: labels.index(x))
print(data.info())

# 4. 提取标签和特征矩阵，分测试集和训练集
X = data.iloc[:, data.columns != "Survived"]
y = data.iloc[:, data.columns == "Survived"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)

# 修正测试集和训练集的索引
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

# 5. 导入模型，粗略跑一下查看结果
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(Xtrain, Ytrain)
score_ = clf.score(Xtest, Ytest)
print(score_)


parameters = {'splitter':('best','random')
                ,'criterion':("gini","entropy")
                ,"max_depth":[*range(1,10)]
                ,'min_samples_leaf':[*range(1,50,5)]
                ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
}
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(Xtrain,Ytrain)
print(GS.best_params_)
print(GS.best_score_)