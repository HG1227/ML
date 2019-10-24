#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 4_kNN 解决鸢尾花和手写数字识别分类问题.py
"""

from sklearn import datasets
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt

def test1_yuan():
    '''
    该数据集包含150个样本和4个数值特征，包括叶片萼片等
    标签y是三种鸢尾花的类别
    :return: score int
    '''
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=321)
    KNN_clf=KNeighborsClassifier(n_neighbors=3)
    KNN_clf.fit(X_train,y_train)
    y_predict=KNN_clf.predict(x_test)
    sc=accuracy_score(y_test,y_predict)

    print(sc)

def test_Digital_recognition():
    '''
    手写数字识别数据集
    数据集包括1797个样本64个特征（8*8）
    标签y是[0,9] 10个数字
    :return:
    '''
    digits=datasets.load_digits()
    X=digits.data
    print(X.shape)      #(1797, 64)
    y=digits.target

    #随机查看一个数字
    print(X[321])
    some_digit=X[321].reshape(8,8)
    plt.imshow(some_digit,cmap=matplotlib.cm.binary)
    plt.show()

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=321)
    KNN_clf=KNeighborsClassifier(n_neighbors=3)
    KNN_clf.fit(X_train,y_train)
    sc=KNN_clf.score(X_test,y_test)
    print(sc)

    y_predict=KNN_clf.predict(X_test)
    ssc=accuracy_score(y_test,y_predict)
    print(ssc)




if __name__=="__main__":
    # test1_yuan()

    test_Digital_recognition()