# TRANSFORMER-INTRO
变形金刚？All you need is Attention？Encode？Decoder？Let's figure it out !

## 模型结构

<img src=".\figs\structure.png" align="middle" alt="structure" style="zoom:70%;" />



模型分为`Encoder`(编码器)和`Decoder`(解码器)两个部分，分别对应上图中的左右两部分。

其中编码器由N个相同的层堆叠在一起(后面的代码取N=6)，每一层又有两个子层：

第一个子层是一个`Multi-Head Attention`(多头的自注意机制)，

第二个子层是一个简单的`Feed Forward`(全连接前馈网络)。

两个子层都添加了一个残差连接（Add） + layer normalization（Norm）的操作。

模型的解码器同样是堆叠了N个相同的层，不过和编码器中每层的结构稍有不同。

对于解码器的每一层，除了编码器中的两个子层`Multi-Head Attention`和`Feed Forward`，

解码器还包含一个子层`Masked Multi-Head Attention`，

如图中所示每个子层同样也用了residual以及layer normalization。

模型的输入由`Input Embedding`和`Positional Encoding`(位置编码)两部分组合而成，

模型的输出由Decoder的输出简单的经过softmax得到。

## 1.Embeeding层



## 2.位置编码



## 3.编码器



## 4.编码器层



## 5.注意力机制



## 6.多头注意力机制



## 7.前馈全连接层



## 8.规范化层



## 9.掩码及其作用



## 10.解码器整体结构



## 11.解码器层



## 12.模型输出



## 13. 模型结构

