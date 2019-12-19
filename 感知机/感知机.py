#!/usr/bin/python
# coding:utf-8
# @software: PyCharm
# @file: 感知机.py
# @time: 2019/12/4

import numpy as np


def makeLinearSeparableData(weights, numLines):
    '''
    numFeatures 是一个正整数，代表特征的数量
    :param weights:是一个列表，里面存储的是我们用来产生随机数据的那条直线的法向量。
    :param numLines:是一个正整数，表示需要创建多少个数据点。
    :return:最后返回数据集合。
    '''
    w = np.array(weights)
    numFeatures = len(weights)
    dataSet = np.zeros((numLines, numFeatures + 1))

    for i in range(numLines):
        x = np.random.rand(1, numFeatures) * 20 - 10
        # 计算内积
        innerProduct = np.sum(w * x)
        if innerProduct <= 0:
            # numpy 提供的 append 函数可以扩充一维数组，
            dataSet[i] = np.append(x, -1)
        else:
            dataSet[i] = np.append(x, 1)
    return dataSet


# 将数据集可视化
def plotData(dataSet):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Linear separable data set')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    labels = np.array(dataSet[:, 2])

    # where 函数是用来找出正例的行的下标
    idx_1 = np.where(dataSet[:, 2] == 1)
    p1 = ax.scatter(dataSet[idx_1, 0], dataSet[idx_1, 1], marker='o',
                    c='g', s=20, label=1)

    idx_2 = np.where(dataSet[:, 2] == -1)
    p2 = ax.scatter(dataSet[idx_2, 0], dataSet[idx_2, 1], marker='x',
                    color='r', s=20, label=2)

    plt.legend(loc='upper right')
    plt.show()





# 训练感知机，可视化分类器及其法向量
def train(dataSet, plot=False):
    ''' (array, boolean) -> list
    Use dataSet to train a perceptron
    dataSet has at least 2 lines.
    '''

    # 随机梯度下降算法
    numLines = dataSet.shape[0]
    numFearures = dataSet.shape[1]
    w = np.zeros((1, numFearures - 1))  # initialize weights

    separated = False
    i = 0
    alpha = 0.5
    while not separated and i < numLines:
        if dataSet[i][-1] * np.sum(w * dataSet[i, 0:-1]) <= 0:  # 如果分类错误
            w = w + alpha * dataSet[i][-1] * dataSet[i, 0:-1]  # 更新权重向量
            separated = False  # 设置为未完全分开
            i = 0  # 重新开始遍历每个数据点
        else:
            i += 1  # 如果分类正确，检查下一个数据点

    if plot == True:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Linear separable data set')
        plt.xlabel('X')
        plt.ylabel('Y')
        labels = np.array(dataSet[:, 2])

        idx_1 = np.where(dataSet[:, 2] == 1)
        p1 = ax.scatter(dataSet[idx_1, 0], dataSet[idx_1, 1],
                        marker='o', color='g', label=1, s=20)

        idx_2 = np.where(dataSet[:, 2] == -1)
        p2 = ax.scatter(dataSet[idx_2, 0], dataSet[idx_2, 1],
                        marker='x', color='r', label=2, s=20)

        # 为了避免求得的权重向量长度过大在散点图中无法显示，所以将它按比例缩小了。
        x = w[0][0] / np.abs(w[0][0]) * 10
        y = w[0][1] / np.abs(w[0][0]) * 10

        ys = (-12 * (-w[0][0]) / w[0][1], 12 * (-w[0][0]) / w[0][1])
        ax.add_line(Line2D((-12, 12), ys, linewidth=1, color='blue'))
        plt.legend(loc='upper right')
        plt.legend(loc='upper right')
        plt.savefig('pe.png',dpi=600)

        plt.show()

    return w

if __name__ == '__main__':
    data = makeLinearSeparableData([4, 3], 100)
    # print(data)
    w = train(data, True)
