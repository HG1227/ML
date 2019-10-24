#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 6_模型参数_n_neighbors.py
"""

# 第二个超参数：n_neighbors
'''
n_neighbors 即要选择最近几个点，默认值是 5（等效 k )。

之前的葡萄酒数据集实际测试下，k 值分别选择 3 和 5 时
k = 3 时，模型得分 0.76，k=5 时模型得分只有 0.68。
所以 k =3 是更好的选择，但可能还存在其他 k 值比 3 效果更好，
怎么才能找到最好的 k 值呢？
最简单的方式就是递归搜索，从中选择得分最高的 k 值。
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


# 设置一个 k 值范围，把刚才的代码封装到 for 循环中来尝试下：
best_k = 0
best_score = 0
for k in range(1, 10):
    KNN_clf = KNeighborsClassifier(k)
    KNN_clf.fit(X_train, y_train)
    score = KNN_clf.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k


print("best_k:",best_k)
print("best_score:",best_score)
# best_k: 3
# best_score: 0.7592592592592593


'''
可以看到，当 k 取 1-10，建立的 10 个模型中，k =3 时的模型得分最高。
这样我们就找到了合适的超参数 k。
'''