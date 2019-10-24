#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 7_模型参数_p.py
"""
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


'''
第四个超参数：p说到距离，还有一个重要的超参数 p。
如果还记得的话，之前的模型在计算距离时，采用的是欧拉距离：
除了欧拉距离，还有一种常用的距离，曼哈顿距离：

这两种距离很好理解，举个例子，从家到公司，欧拉距离就是二者的直线距离，
但显然不可能以直线到公司，而只能按着街道线路过去，
这就是曼哈顿距离，俗称出租车距离。


明可夫斯基距离（Minkowski Distance），曼哈顿距离和欧拉距离分别是 p 为 1 和 2 的特殊值。

使用不同的距离计算公式，点的分类很可能不一样导致模型效果也不一样。
'''
'''
在 Sklearn 中对应这个距离的超参数是 p，默认值是 2，也就是欧拉距离，
下面我们尝试使用不同的 p 值建立模型，看看哪个 p 值得到的模型效果最好，
 这里因为考虑了 p，那么 weights 超参数需是 distance，外面嵌套一个 for 循环即可：
'''

best_k = 0
best_p = 0
best_score = 0

for k in range(1, 20):
    for p in range(1, 10):
        KNN_clf = KNeighborsClassifier(n_neighbors=k, weights='distance', p=p)
        KNN_clf.fit(X_train, y_train)
        score = KNN_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_p = p
            best_score = score
print('best_p:', best_p)
print('best_k:', best_k)
print('best_score:', best_score)
'''
best_p: 1
best_k: 15
best_score: 0.8148148148148148
'''

'''
可以看到，找出的最好模型，超参数 k 值为 15 ，p 值为 1 即曼哈顿距离，模型得分 0.81。
比我们最初采用的超参数：k=3、weights = uniform 得到的 0.76 分要好。
但这里要注意 k=15 很可能是过拟合了，这样即使得分高这个模型也是不好的，以后会说。
'''