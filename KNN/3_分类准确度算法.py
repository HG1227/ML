#!/usr/bin/python
# coding:utf-8

'''
手写分类准确度算法
'''


def accuracy_score(y_test, y_predict):
    assert y_test.shape[0] == y_predict.shape[0], "预测的y值和测试集的y值行数要一样"
    return sum(y_test == y_predict) / len(y_predict)


'''
Sklearn 的第二种方法是直接调用 model.score 方法得到模型分数，
我们仍然可以尝试做到。
'''


def predict(X_test):
    pass


def score(X_test, y_test):
    y_predict = predict(X_test)
    return accuracy_score(y_test, y_predict)

