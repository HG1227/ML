# -*- encoding: utf-8 -*-
"""
@software: PyCharm
@file : Gradient Descent.py
@time : 2019/12/19 
"""

import numpy as np
import matplotlib.pyplot as plt

# Size of the points dataset.
m = 20

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

# The Learning Rate alpha.
alpha = 0.01

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(diff.T, diff)

def gradient_function(theta, X, y):
    '''损失函数求偏导'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(X.T, diff)

def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    # theta 初始化的值
    theta = np.array([1, 1]).reshape(2, 1)
    # gradient 损失函数求偏导后的结果
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        # 更新theta
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])

x_v = np.linspace(1, 21, 1000)
# 拟合直线
y_v = optimal[0][0] + optimal[1][0]*x_v

plt.scatter(X1.reshape(1, -1),y.reshape(1,-1),)
plt.plot(x_v, y_v, color = "r")
plt.savefig("GD.png", dpi = 600)
plt.show()