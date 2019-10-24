#!/usr/bin/python
#coding:utf-8

"""
@software: PyCharm
@file: 5_kNN 模型_调参.py
"""
'''
超参数:就是在模型运行前就要先确定的参数。
模型参数:是在算法运行过程自己学习的参数


kNN 模型的超参数有许多，可以在 sklearn 官网中找到，这里介绍几个最重要的。
第一个超参数：algorithm
    algorithm 即算法，意思就是建立 kNN 模型时采用什么算法去搜索最近的 k 个点，有四个选项：
        * brute（暴力搜索）
        * kd_tree（KD树）
        * ball_tree（球树）
        * auto（默认值，自动选择上面三种速度最快的）

KNN 模型的核心思想是计算大量样本点之间的距离。


第一种算法 brute （暴力搜索）。
意思就是计算预测样本和全部训练集样本的距离，
最后筛选出前 K 个最近的样本，这种方法很蛮力，所以叫暴力搜索。
当样本和特征数量很大的时候，每计算一个样本距离就要遍历一遍训练集样本，
很费时间效率低下。有什么方法能够做到无需遍历全部训练集就可以快速找到需要的
 k 个近邻点呢？这就要用到 KD 树和球树算法。


第二种算法 KD 树（K-dimension tree缩写）。
简单来说 KD 树是一种「二叉树」结构，就是把整个空间划分为特定的几个子空间，
然后在合适的子空间中去搜索待预测的样本点。采用这种方法不用遍历全部样本就可以
快速找到最近的 K 个点，速度比暴力搜索快很多（稍后会用实例对比一下）。至于什
么是二叉树，这就涉及到数据结构的知识，稍微有些复杂，就目前来说暂时不用深入了解，
sklearn 中可以直接调用 KD 树，很方便。之后会单独介绍二叉树和 KD 树，再来手写 
KD 树的 Python 代码。

什么样的数据集适合使用 KD 树算法
假设数据集样本数为 m，特征数为 n，则当样本数量 m 大于 2 的 n 次方时，用 KD 树算
法搜索效果会比较好。比如适合 1000 个样本且特征数不超过 10 个（2 的 10 次方为
1024）的数据集。一旦特征过多，KD 树的搜索效率就会大幅下降，最终变得和暴力搜索差
不多。通常来说 KD 树适用维数不超过 20 维的数据集，超过这个维数可以用球树这种算法。

第三种算法是球树（Ball tree）。
对于一些分布不均匀的数据集，KD 树算法搜索效率并不好，为了优化就产生了球树这种算法。
同样的，暂时先不用具体深入了解这种算法。



'''

from  sklearn import  datasets
from  sklearn.model_selection import train_test_split
from  sklearn.neighbors import KNeighborsClassifier
import time

#第一个超参数：algorithm
#第一个数据集是样本数 m 大于样本 2 的 n 次方，依次建立 kNN 模型并计算运行时间：
def test1(m,n):
    # m=2000
    # n=10
    data=datasets.make_classification(n_samples=m,n_features=n,n_classes=2,random_state=321)

    X=data[0]
    print(X.shape)
    y=data[1]
    print(y.shape)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=321)

    #1 brute 暴力搜索法

    KNN_clf=KNeighborsClassifier(algorithm='brute',n_jobs=-1)   #n_jobs=-1表示使用电脑全部核进行运算
    s=time.time()
    KNN_clf.fit(X_train,y_train)
    sc=KNN_clf.score(X_test,y_test)
    print("SC",sc)
    end=time.time()
    print("brute=",end-s)

    #2 KD 树

    KNN_clf_KD=KNeighborsClassifier(algorithm='kd_tree',n_jobs=-1)
    s=time.time()
    KNN_clf_KD.fit(X_train,y_train)
    sc_kd=KNN_clf_KD.score(X_test,y_test)
    print('sc_kd:',sc_kd)
    end=time.time()
    print("KD=",end-s)


    #3 球 树
    KNN_clf_ball_tree=KNeighborsClassifier(algorithm='ball_tree',n_jobs=-1)
    s=time.time()
    KNN_clf_ball_tree.fit(X_train,y_train)
    sc_ball_tree=KNN_clf_ball_tree.score(X_test,y_test)
    print('sc_ball_tree:',sc_ball_tree)
    end=time.time()
    print("ball_tree=",end-s)


    #3 auto 自动选择
    KNN_clf_auto=KNeighborsClassifier(algorithm='auto',n_jobs=-1)
    s=time.time()
    KNN_clf_auto.fit(X_train,y_train)
    sc_auto=KNN_clf_auto.score(X_test,y_test)
    print('sc_auto:',sc_auto)
    end=time.time()
    print("auto=",end-s)

    '''
    *可见当数据集样本数量 m > 2 的 n  次方时，kd_tree 和 ball_tree 速度比 brute 暴力搜索快了一个量级，
        auto 采用其中最快的算法。
    *当数据集样本数量 m 小于 2 的 n 次方 时，KD 树和球树的搜索速度大幅降低，暴力搜索算法相差无几。
    *我们介绍了第一个超参数 algorithm，就 kNN 算法来说通常只需要设置 auto 即可，让模型自动为我们选择合适的算法。
    '''







if __name__=="__main__":
    m1=20000
    n1=10
    test1(m1,n1)

    m1=20000
    n1=20
    test1(m1,n1)
