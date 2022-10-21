# 数据预处理
针对一个数据集。合理的数据预处理可以为之后的任务带来诸多方便。
[POS任务数据集](https://zenodo.org/record/4835806#.YsLafWBBzSJ)
可以用open('',..,..)方式打开文件。读取行不定长的文本文件




# word to vector
**word2vector**
A way to do word embedding
将“不可计算”“非结构化”的词转化为“可计算”“结构化”的向量。

|训练模式|解释||
|--|--|--|
|CBOW(Continuous Bag-of-Words Model)和|||
|Skip-gram (Continuous Skip-gram Model)||



|训练技巧|trick|
|--|--|
|hierarchical softmax|本质是把 N 分类问题变成 log(N)次二分类|
|negative sampling|本质是预测总体类别的一个子集|


# seq2seq







skip-gram: window

# 不定长


# masked LM
否则会出现自己预测自己。理论上预测时候是不知道下一个输出是什么的。
也会出现很多有趣的 mask 方法

# seq2seq模型(Sequence to Sequence模型)


