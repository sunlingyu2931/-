### K-means

下面会说三种聚类方法

# 一：概念

一种无监督算法。

规定要聚类的个数，也就是k。然后计算每个点到k的距离，欧几里得方法，分类到距离近的k个中心点之一，然后再计算每个类别中新的中心点坐标。

再次进行所有点到中心点的距离，归类到近的中心点，更新中心点坐标，一直循环，知道中心点的坐标不再 变化。



```
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/sunlingyu/Downloads/dataset python self-study/kmeans.txt')
k =4

model = KMeans(n_clusters=k) # define model
model.fit(data)  # 传入数据

centers  = model.cluster_centers_    # 中心点/质心的坐标
print(centers)

result = model.predict(data)
print(result)

mark = ['or','ob','og','oy'] # o是形状 rbgy是颜色
for i,d in enumerate(data):    #i index    d data
    plt.plot(d[0],d[1],mark[result[i]])

mark = ['*r','*b','*g','*y']
for i,center in enumerate(centers):
    plt.plot(center[0],center[1],mark[i],markersize=20)
plt.show()
```



#### enumerate() 函数

- 进行遍历，enumerate将其组成一个索引序列，利用它可以同时获得索引和值
- enumerate多用于在for循环中得到计数



```
for index, item in enumerate(list):
     print(index, item)    # for后面第一个定义索引（从0开始），第二个数据的名字
```





### mini batch k-means

数据量大的时候可以用，Mini Batch从不同类别的样本中抽取一部分样本来代表各自类型进行计算。由于计算样本量少，所以会相应的减少运行时间，但另一方面抽样也必然会带来准确度的下降。其他的和普通的k-means一样

Code 也和kmeans一样，只是import 包的时候变一下

```
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/sunlingyu/Downloads/dataset python self-study/kmeans.txt')
k =4

model = MiniBatchKMeans(n_clusters=k)
model.fit(data)

centers = model.cluster_centers_
print(centers)

result = model.predict(data)
print(result)

mark = ['or','ob','og','oy'] # o是形状 rbgy是颜色
for i,d in enumerate(data):    #i index    d data
    plt.plot(d[0],d[1],mark[result[i]])

mark = ['*r','*b','*g','*y']
for i,center in enumerate(centers):
     plt.plot(center[0],center[1],mark[i],markersize=20)
     plt.show()
```



# 二：优缺点

1.初始质心的选取也比较重要

2.不同的k值的选取会带来不同的结果：肘部法则

2.对于非球形的数据据效果并不好



# 三：cost function

![Screen Shot 2019-06-12 at 10.07.20 am](/Users/sunlingyu/Desktop/Screen Shot 2019-06-12 at 10.07.20 am.png)



x就是每个点，u就是x那个点所对应的质心

我们用sklearn包的话就已经包括这个过程了，不需要再考虑



# 四：Density-based spatial clustering of application with noise DBSCAN 基于密度的方法

epsilon 就是半径的大小

minponits 就是只要一个范围大于minpoints这个数字就会成为一个簇

这个方法不需要自己设置簇的个数





https://www.naftaliharris.com/blog/visualizing-k-means-clustering/

上面这个网址可以帮助理解这个过程





```
example 1
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/sunlingyu/Downloads/dataset python self-study/kmeans.txt',delimiter=' ')
# print(data)
model = DBSCAN(eps = 1,min_samples=4)
result = model.fit_predict(data) # fit和predict一起做
print(result)

mark = ['or','ob','og','oy','ok','om']
for i,d in enumerate(data):
    plt.plot(d[0],d[1],mark[result[i]])
plt.show()



example 2
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

x1,y1 = datasets.make_circles(n_samples=2000,factor=0.5,noise=0.05) # 生成数据， circle 圆形的
x2,y2 = datasets.make_blobs(n_samples=1000,centers=[[1.2,1.2]],cluster_std=[[0.1]]) # 中心点(1.2,1.2)
x = np.concatenate((x1,x2))
plt.scatter(x[:,0],x[:,1],marker='o')
# plt.show()  #kmeans 

from sklearn.cluster import KMeans # 用kmeans试一下
y_pred = KMeans(n_clusters=3).fit_predict(x)
plt.scatter(x[:,0],x[:,1],c=y_pred) # c=y_pred color根据标签来表示
plt.show()

from sklearn.cluster import DBSCAN
y_pred1 = DBSCAN(eps=0.2,min_samples=50).fit_predict(x)  # 传入参数并进行预测
plt.scatter(x[:,0],x[:,1],c=y_pred1)
plt.show()  #dbscan
```

