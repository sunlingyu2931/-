### decision tree 决策树

（既可以classification也可以进行 regression，取决于特征的选择）



知识储备：

1. 信息熵（entropy）：表示random variable的不确定性

概率等于0.5的时候不确定性是最高的。

信息熵的公式log是base为2的，所以由于概率是0-1的，log的值是负数。



2. 信息：帮助消除不确定性的事物

（不能帮助人们消除不确定性的事物称为噪音或者数据）



3. 信息增益：特征x使得类别y的不确定性减少的程度







一.  Definition

一颗由多个判断节点组成的树，可以是二叉树或者非二叉树。



重要的组成部分：root node根节点，non-leaf node 非叶子节点，leaf node 叶子节点

（leaf node和non-leaf node的区别是leaf node不可以继续往下分了）



那如何决定特征呢？

计算entropy，再计算gain（信息增益），选择information gain大的作为特征

（分类特征一般要将连续型数据转为分散数据，类似分段函数那种）





二.类别

ID3

C4.5

CART (classification and regression trees)

这三种方法道理都差不多，只是有些标准不同。



