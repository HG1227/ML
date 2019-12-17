# -*- encoding: utf-8 -*-
"""
@software: PyCharm
@file : PCA.py
@time : 2019/12/17 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 计算均值，要求输入数据为 numpy 的矩阵格式，行表示样本数，列表示特征
def mean(data):
    # axis = 0表示按照列来求均值,如果输入list,则axis=1
    return np.mean(data, axis=0)


def pca(XMat, k):
    '''
    :param XMat:传入的是一个numpy的矩阵格式，行表示样本数，列表示特征
    :param k: 表示取前k个特征值对应的特征向量
    :return:
    	- finalData：参数一指的是返回的低维矩阵，对应于输入参数二
	    - reconData：参数二对应的是移动坐标轴后的矩阵
    '''
    average = mean(XMat)
    m, n = np.shape(XMat)
    # data_adjust 减去均值之后的矩阵
    data_adjust = XMat - average

    cov_mat = np.cov(data_adjust.T)  # 计算协方差矩阵
    feat_value, feat_vec = np.linalg.eig(cov_mat)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-feat_value)  # 按照featValue进行从大到小排序

    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.mat(feat_vec.T[index[:k]])   # 最大的k个特征值对应的特征向量构成的矩阵
        final_data = data_adjust * selectVec.T      # 低维特征空间的数据(将样本点投影到选取的特征向量上)
                                                    # 降维后的矩阵，相当于 sklearn 中 PCA transform后的返回值

        reconData = (final_data * selectVec) + average  # 重构数据(不太明白)
    return final_data, reconData


def plotBestFit(finalData, reconMat):
    finalData = np.array(finalData)
    reconMat = np.array(reconMat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(finalData[:, 0], finalData[:, 1], s=50, c='red', marker='s', label="finalData")
    ax.scatter(reconMat[:, 0], reconMat[:, 1], s=50, c='blue', label="reconMat")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = np.array(pd.read_csv('data.csv', index_col=None, header=None))
    k = 2
    finalData, reconMat = pca(data, k)
    print(finalData)
    plotBestFit(finalData, reconMat)
