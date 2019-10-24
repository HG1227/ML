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
# print(x)

'''
[[  1.           0.        ]
 [  1.           0.        ]
 [  2.           0.01010101]
 [  2.           0.01010101]
 [  3.           0.02020202]
 [  3.           0.02020202]
 [100.           1.        ]]
'''


#Sklearn 最值归一化
from  sklearn import  datasets
from  sklearn.model_selection import train_test_split

wine=datasets.load_wine()
# print(wine.DESCR)
X=wine.data
y=wine.target
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.3,
                                               random_state=321)

'''
现在，将全部特征映射到 0-1 的范围后建模，预测模型得分效果如何，分三个步骤。
第一步，将数据归一化。加载和划分数据集已经练习过很多遍了不用多说。Sklearn 中的最
值归一化方法是 MinMaxScaler 类，fit  拟合之后调用 fit_transform 方法归一化训练集。
'''

#导入归一化方法并fit
from sklearn.preprocessing import MinMaxScaler
minmaxscaler=MinMaxScaler()
#MinMaxScaler(feature_range=(0, 1), copy=True)

minmaxscaler.fit(X_train)

#fit之后才可以transform 归一化
X_train_minmax=minmaxscaler.fit_transform(X_train)


'''
解释一下 fit 、 fit_transform 和 transform 的作用和关系。
在执行归一化操作时，一定要先 fit 然后才能调用 fit_transform。因为 fit 的作用是计
算出训练集的基本参数，比如最大最小值等。只有当最大最小值有确定的数值时，才能通
过 fit_tranform 方法的断言，进而才能运行并进行归一化操作。否则不先运行 fit 那么
最大最小值的值是空值，就通不过 fit_transform 的断言，接着程序就会报过
fit_tranform 方法的断言，进而才能运行并进行归一化操作。否则不先运行 fit 那么错。
虽然计算最大最小值和归一化这两个步骤不相关，但 Sklearn 的接口就是这样设计的。

所以，要先 fit 然后 fit_transform。至于 transform，它的作用和 fit_transform 差不
多，只是在不同的方法中应用。比如 MinMaxScaler 中是 fit_transform，等下要讲的
均值方差归一化 StandardScaler 中则是 transform。
'''


#第二步，建立 kNN 模型。这一步也做过很多遍了。
#建立KNN模型
from  sklearn.neighbors import  KNeighborsClassifier

KNN_clf=KNeighborsClassifier(n_neighbors=3)
KNN_clf.fit(X_train_minmax,y_train)

#第三步，测试集归一化并预测模型。
#这里一定要注意测试集归一化的方法：是在训练集的最大最小值基础归一化，而非测试集的最大最小值。
'''
为什么是在训练集上呢？其实很好理解，虽然我们划分出的测试集能很容易求出最大最小值，但是别忘了我们划分测试集的目的：来模拟实际中的情况。而在实际中，通常很难获得数据的最大最小值，因为我们得不到全部的测试集。比如早先说的案例：酒吧猜新倒的一杯红酒属于哪一类。凭这一杯葡萄酒它是没有最大最小值的，你可能说加多样本，多几杯红酒不就行了？但这也只是有限的样本，有限样本中求出的最大最小值是不准确的，因为如果再多加几杯酒，那参数很可能又变了。
所以，测试集的归一化要利用训练集得到的参数，包括下面要说的均值方差归一化。
'''

#注意两点  1 在预测前也必须对测试集归一化
#         2 归一化采用的训练集的最大最小值，而不是测试集

X_test_minmax=minmaxscaler.fit_transform(X_test)

#计算测试集得分
sc=KNN_clf.score(X_test_minmax,y_test)
print(sc)       #0.9074074074074074


#作为对比  求出未归一化的得分
KNN_clf=KNeighborsClassifier(n_neighbors=3)

KNN_clf.fit(X_train,y_train)
sc=KNN_clf.score(X_test,y_test)
print(sc)