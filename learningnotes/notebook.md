# 如何解决过拟合、欠拟合【学习总结】

## 一、过拟合(Overfitting)
### 1. 概念：
If you can fit the training data,but largeerror on testing data,then you probably have large variance.
过拟合是指模型在训练集上的表现很好，但在测试集和新数据上的表现很差。

### 2. 特征：
*Small Bias; Large Variance；Complex Model.*

### 3.一般解决思路：
**（1）More Data:** 使用更多的训练数据是解决过拟合问题最有效的手段，因为更多的样本能够让模型学习到更多有效的特征，减少噪声的影响。

**（2）Regularizationg:** 给模型参数添加一定的正则约束。

**（3）降低模型复杂度:** 适当降低模型复杂度可以避免模型拟合过多的采样噪声。

**（4）使用集成学习方法:** 把多个模型集成在一起，降低单一模型的过拟合风险，如Bagging。
<br>


## 二、欠拟合(Underfitting)
### 1. 概念：
If your model cannot even fit the training examples,then you have large bias.
欠拟合是指训练的模型不能很好地拟合数据关系。

### 2. 特征：
*Large Bias; Small Variance; Simple Model.*

### 3. 一般解决思路：
**（1） Redesign Model:** 重新设计模型，增加模型的复杂度。简单模型的学习能力较差，增加模型复杂度可以使模型有更强的拟合能力。
 例如，在线性模型中添加高次项，在神经网络模型中增加网络层数或神经元个数等。
 
**（2）Add more features as input:** 当特征不足或现有特征与样本标签的相关性不强时，模型容易出现欠拟合。



*参考文献：学习总结参考李老师公开课。*
