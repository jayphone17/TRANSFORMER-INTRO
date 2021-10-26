"""
transformer 网络结构
"""
import math
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import LayerNorm

"""
Embedding层

Embedding层的作用是将某种格式的输入数据，

例如文本，转变为模型可以处理的向量表示，

来描述原始数据所包含的信息。

Embedding层输出的可以理解为当前时间步的特征，

如果是文本任务，这里就可以是Word Embedding，

如果是其他任务，就可以是任何合理方法所提取的特征。

核心是借助torch提供的nn.Embedding
"""
class Embedding(nn.Module):
    def __init__(self,d_model,vocab):
        # 类的初始化函数
        # d_model：指词嵌入的维度
        # vocab: 指词表的大小
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        # 调用nn中的预定义层Embedding，获得一个词嵌入对象self.lut
        self.d_model = d_model
        # 最后就是将d_model传入类中
    def forward(self,x):
        # Embedding层的前向传播逻辑
        # 参数x：这里代表输入给模型的单词文本通过词表映射后的one - hot向量
        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        embedds = self.lut(x)
        return embedds*math.sqrt(self.d_model)

"""
位置编码 PositionalEncoding

Positional Encodding位置编码的作用是为模型提供当前时间步的前后出现顺序的信息。

因为Transformer不像RNN那样的循环结构有前后不同时间步输入间天然的先后顺序，

所有的时间步是同时输入，并行推理的，因此在时间步的特征中融合进位置编码的信息是合理的。

位置编码可以有很多选择，可以是固定的，也可以设置成可学习的参数。

我们使用固定的位置编码,使用不同频率的sin和cos函数来进行位置编码

可以认为，最终模型的输入是若干个时间步对应的embedding，

每一个时间步对应一个embedding，可以理解为是当前时间步的一个综合的特征信息，

即包含了本身的语义信息，又包含了当前时间步在整个句子中的位置信息
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        # d_model：词嵌入维度
        # dropout: dropout触发比率
        # max_len：每个句子的最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        # Compute the positional encodings
        # 注意下面代码的计算方式与公式中给出的是不同的，但是是等价的，你可以尝试简单推导证明一下。
        # 这样计算是为了避免中间的数值计算结果超出float的范围

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 2),*-(math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


"""
编码器 Encoder

编码器作用是用于对输入进行特征提取，为解码环节提供有效的语义信息
"""
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 调用时会将编码器层传进来，我们简单克隆N分，叠加在一起，组成完整的Encoder
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



"""
编码器层

每个编码器层由两个子层连接结构组成
第一个子层包括一个多头自注意力层和规范化层以及一个残差连接
第二个子层包括一个前馈全连接层和规范化层以及一个残差连接
"""

"""
注意力机制 attention
"""

"""
多头注意力机制
"""


"""
前馈全连接层
"""


"""
规范化层
"""


"""
掩码及其作用
"""


"""
解码器
"""


"""
解码器层
"""

"""
模型输出
"""


"""
整体模型构建
"""