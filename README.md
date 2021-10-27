# TRANSFORMER-INTRO
***变形金刚？All you need is Attention？Encoder？Decoder？Let's figure it out !***

## 模型结构

<img src=".\figs\structure.png" align="middle" alt="structure" style="zoom:70%;" />



```
模型分为`Encoder`(编码器)和`Decoder`(解码器)两个部分，分别对应上图中的左右两部分。

其中编码器由N个相同的层堆叠在一起(后面的代码取N=6)，每一层又有两个子层：

第一个子层是一个`Multi-Head Attention`(多头的自注意机制)，

第二个子层是一个简单的`Feed Forward`(全连接前馈网络)。

两个子层都添加了一个残差连接（Add） + layer normalization（Norm）的操作。

模型的解码器同样是堆叠了N个相同的层，不过和编码器中每层的结构稍有不同。

对于解码器的每一层，除了编码器中的两个子层`Multi-Head Attention`和`Feed Forward`，

解码器还包含一个子层`Masked Multi-Head Attention`，

如图中所示每个子层同样也用了residual以及layer normalization。

模型的输入由`Input Embedding`和`Positional Encoding`(位置编码)两部分组合而成，

模型的输出由Decoder的输出简单的经过softmax得到。
```



## 1.Embeeding层

```
Embedding层的作用是将某种格式的输入数据，

例如文本，转变为模型可以处理的向量表示，

来描述原始数据所包含的信息。

Embedding层输出的可以理解为当前时间步的特征，

如果是文本任务，这里就可以是Word Embedding，

如果是其他任务，就可以是任何合理方法所提取的特征。

核心是借助torch提供的nn.Embedding
```

## 2.位置编码

```
位置编码 PositionalEncoding

Positional Encodding位置编码的作用是为模型提供当前时间步的前后出现顺序的信息。

因为Transformer不像RNN那样的循环结构有前后不同时间步输入间天然的先后顺序，

所有的时间步是同时输入，并行推理的，因此在时间步的特征中融合进位置编码的信息是合理的。

位置编码可以有很多选择，可以是固定的，也可以设置成可学习的参数。

我们使用固定的位置编码,使用不同频率的sin和cos函数来进行位置编码
```

<img src=".\figs\gongshi.png" align="middle" alt="gongshi" style="zoom:75%;" />

```
可以认为，最终模型的输入是若干个时间步对应的embedding，

每一个时间步对应一个embedding，可以理解为是当前时间步的一个综合的特征信息，

即包含了本身的语义信息，又包含了当前时间步在整个句子中的位置信息
```

## 3.编码器

```
编码器作用是用于对输入进行特征提取，为解码环节提供有效的语义信息
```

## 4.编码器层

```
每个编码器层由两个子层连接结构组成
第一个子层包括一个多头自注意力层和规范化层以及一个残差连接
第二个子层包括一个前馈全连接层和规范化层以及一个残差连接
```

## 5.注意力机制

```
人类在观察事物时，无法同时仔细观察眼前的一切，只能聚焦到某一个局部。

通常我们大脑在简单了解眼前的场景后，能够很快把注意力聚焦到最有价值的局部来仔细观察，

从而作出有效判断。或许是基于这样的启发，想到了在算法中利用注意力机制。

注意力计算：它需要三个指定的输入Q（query），K（key），V（value），然后通过下面公式得到注意力的计算结果。
```
<img src=".\figs\gongshi2.png" align="middle" alt="structure"  />

<img src=".\figs\4.png" align="middle" alt="structure"  />

```
当前时间步的注意力计算结果，是一个组系数 * 每个时间步的特征向量value的累加，

而这个系数，通过当前时间步的query和其他时间步对应的key做内积得到，

这个过程相当于用自己的query对别的时间步的key做查询，判断相似度，

决定以多大的比例将对应时间步的信息继承过来
```

## 6.多头注意力机制



## 7.前馈全连接层



## 8.规范化层



## 9.掩码及其作用



## 10.解码器整体结构



## 11.解码器层



## 12.模型输出



## 13. 模型结构

