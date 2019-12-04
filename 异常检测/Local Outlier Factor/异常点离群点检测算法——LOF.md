# 					异常点/离群点检测算法——LOF

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/bfdd5e2ely1fweovqontkj20ro0rojtj.jpg)

## 1. 算法介绍

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/%E5%B9%BB%E7%81%AF%E7%89%871.PNG)

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/%E5%B9%BB%E7%81%AF%E7%89%872.PNG)

![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/%E5%B9%BB%E7%81%AF%E7%89%873.PNG)



##  2. `sklearn` 模块**中的`LocalOutlierFactor `(LOF)

```python 
class sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, 
                                           algorithm=’auto’,
                                           leaf_size=30, 
                                           metric=’minkowski’, 
                                           p=2, 
                                           metric_params=None, 
                                           contamination=0.1, 
                                           n_jobs=1)

```

**1）主要参数**

- `n_neighbors` :  设置k，default=20
- `contamination` :  设置样本中异常点的比例，范围为 (0, 0.5)，表示样本中的异常点比例，默认为 0.1

**2）主要函数：**

-  `fit_predict(X)`: 

   无监督学习，只需要传入训练数据data，传入的数据维度**至少是 2 维**						

  返回一个数组，-1表示异常点，1表示正常点。

- `clf._decision_function(data)` :  获取每一个样本点的 LOF 值，该函数获取 LOF 值的相反数，需要取反号。
  **`clf._decision_function` 的输出方式更为灵活：**若使用` clf._predict(data) `函数，则按照原先设置的 contamination 输出判断结果（按比例给出判断结果，异常点返回-1，非异常点返回1）

**3）主要属性：**

-  `negative_outlier_factor_ `:  返回值 numpy array, shape (n_samples,)   

  训练样本的异常性	和LOF相反的值，值越小，越有可能是异常点。
  					（注：上面提到LOF的值越接近1，越可能是正常样本，LOF的值越大于1，则越可能是异常样本）。这里就正好反一下。

  ​	

### LOF实例（sklearn）

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor


np.random.seed(42)

# Generate train data
X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

# Generate some outliers
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_

plt.title("Local Outlier Factor (LOF)")
plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
legend = plt.legend(loc='upper left')

# 重新设置图例符号的大小
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
```
![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/20191204230501.png)


参考：<scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-outlier-detection-py>
$$
\begin{equation}
1+2+3+\dots+(n-1)+n = \frac{n(n+1)}{2}
\end{equation}
$$
​     ![](https://raw.githubusercontent.com/HongGHu/tuchuang/master/00642gdOly1g6fvgc7ivuj30go0go3z1.jpg)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
