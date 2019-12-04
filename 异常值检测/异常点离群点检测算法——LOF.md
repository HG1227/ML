# 					异常点/离群点检测算法——LOF

## **1.  局部异常因子算法-Local Outlier Factor (LOF)**

​		在数据挖掘方面，经常需要在做特征工程和模型训练之前对数据进行清洗，剔除无效数据和异常数据。异常检测也是数据挖掘的一个方向，用于反作弊、伪基站、金融诈骗等领域。
　　异常检测方法，针对不同的数据形式，有不同的实现方法。常用的有基于分布的方法，在上、下α分位点之外的值认为是异常值（例如图1），对于属性值常用此类方法。基于距离的方法，适用于二维或高维坐标体系内异常点的判别，例如二维平面坐标或经纬度空间坐标下异常点识别，可用此类方法。



 **K-邻近距离**（k-distance）：在距离数据点 $p$ 最近的几个点中，第 k 个最近的点跟点 p 之间的距离称为点 p 的 K-邻近距离，记为 k-distance (p) 

**可达距离**（rechability distance）：可达距离的定义跟K-邻近距离是相关的，给定参数k时， 数据点 p 到 数据点 o 的可达距离 reach-dist（p, o）为数据点 o 的K-邻近距离 和 数据点p与点o之间的直接距离的最大。<br>
$$
reach\_dist_k(p,o) = max\{k-distance(o),d(p,o)\}
$$


<br>

##  **2. `sklearn` 模块**中的`LocalOutlierFactor `(LOF)

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

参考：[Outlier detection with Local Outlier Factor (LOF)](scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html#sphx-glr-auto-examples-neighbors-plot-lof-outlier-detection-py)