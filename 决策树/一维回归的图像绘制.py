#!/usr/bin/python
# coding:utf-8

"""
@software: PyCharm
@file: 一维回归的图像绘制.py
"""

# 1. 导入需要的库
import numpy as  np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# 2. 创建一条含有噪声的正弦曲线
rag = np.random.RandomState(1)
X = np.sort(5 * rag.rand(80, 1), axis=0)  # axis=1为横向，axis=0为纵向
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rag.rand(16))

# 3. 实例化&训练模型
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# 4. 测试集导入模型，预测结果
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)


#5. 绘制图像
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()