# -*- encoding: utf-8 -*-
"""
@software: PyCharm
@file : LR.py
@time : 2019/12/20 
"""

import matplotlib.pyplot as plt
import numpy as np


# 加载数据
def load_file(filename):
    file = open(filename)
    x = list()
    y = list()
    for line in file.readlines():
        line = line.strip().split()
        x.append([1, float(line[0]), float(line[1])])
        y.append(float(line[-1]))

    xmat = np.mat(x)
    ymat = np.mat(y).T
    file.close()

    return xmat, ymat


def w_calc(xmat, ymat, alpha=0.001, max_iter=100001):
    '''
    :param xmat: 样本
    :param ymat: 样本标签
    :param alpha: 学习率
    :param max_iter: 最大迭代次数
    :return: 系数矩阵
    '''

    # 初始化W 系数矩阵 三行一列
    W = np.mat(np.random.randn(3, 1))
    # 更新 W
    for i in range(max_iter):
        # sigmoid 函数
        H = 1 / (1 + np.exp((-xmat * W)))
        # 损失函数求偏导后的结果
        dw = xmat.T * (H - ymat)
        W -= alpha * dw
    return W


xmat, ymat = load_file('test.txt')
print(ymat)

# 系数矩阵
W = w_calc(xmat, ymat)
w0 = W[0, 0]
w1 = W[1, 0]
w2 = W[2, 0]

plotx1 = np.arange(2, 6, 0.01)
plotx2 = -(w0 + w1 * plotx1) / w2
# 得到的分类线
plt.plot(plotx1, plotx2)

# .A 将矩阵转化为 array 的形式
plt.scatter(xmat[:, 1][ymat == 0].A, xmat[:, 2][ymat == 0].A)
plt.scatter(xmat[:, 1][ymat == 1].A, xmat[:, 2][ymat == 1].A)
plt.savefig('lr.png',dpi=600)
plt.show()
