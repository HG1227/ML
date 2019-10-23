#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 1_简单实现.py
"""

import matplotlib.pyplot as plt
import numpy as np


def loadfile(filename):
    file = open(filename)
    x = []
    y = []
    for line in file.readlines():
        line = line.strip().split()
        x.append([1, float(line[0]), float(line[1])])
        y.append(float(line[-1]))

    xmat = np.mat(x)
    ymat = np.mat(y).T
    file.close()
    return xmat, ymat


def w_calc(xmat, ymat, alpha=0.001, maxIter=100001):
    # W init         3行  1列
    W = np.mat(np.random.randn(3, 1))

    # W update
    for i in range(maxIter):
        H = 1 / (1 + np.exp(-xmat * W))
        dw = xmat.T * (H - ymat)
        W -= alpha * dw

    return W


xmat , ymat = loadfile('test.txt')

print(xmat)
print(ymat)

# W = w_calc(xmat, ymat, 0.0001, 100001)
W=w_calc(xmat,ymat)
w0 = W[0, 0]
w1 = W[1, 0]
w2 = W[2, 0]

plotx1 = np.arange(2, 6, 0.01)
plotx2 = -w0 / w2 - w1 / w2 * plotx1
plt.plot(plotx1, plotx2)
plt.scatter(xmat[:, 1][ymat == 0].A, xmat[:, 2][ymat == 0].A)
plt.scatter(xmat[:, 1][ymat == 1].A, xmat[:, 2][ymat == 1].A)
plt.show()
