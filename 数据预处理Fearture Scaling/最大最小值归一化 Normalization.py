#!/usr/bin/python
#coding:utf-8

"""
@software: PyCharm
@file: 最大最小值归一化.py
"""
'''
Xscl=(X-Xmin)/(Xmax-Xmin)
实际处理的过程中针对同一列的数据

最大最小归一化不适合极端数据分布
'''
import  numpy as np
x=np.array([1,1,2,2,3,3,100]).reshape(-1,1)

x2=(x-x.min())/(x.max()-x.min())


np.set_printoptions(suppress=True)
x=np.hstack([x,x2])
# x=np.concatenate((x,x2),axis = 1)
print(x)

'''
[[  1.           0.        ]
 [  1.           0.        ]
 [  2.           0.01010101]
 [  2.           0.01010101]
 [  3.           0.02020202]
 [  3.           0.02020202]
 [100.           1.        ]]
'''


