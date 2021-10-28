# 笔记

## 1. Self-Attention

<img src=".\note_figs\image-20211027153126440.png" alt="image-20211027153126440" style="zoom:50%;" />

缺点：看不出关系

<img src=".\note_figs\image-20211027153229726.png" alt="image-20211027153229726" style="zoom:50%;" />

<img src=".\note_figs\image-20211027153315383.png" alt="image-20211027153315383" style="zoom: 50%;" />

一排向量

<img src=".\note_figs\image-20211027153407036.png" alt="image-20211027153407036" style="zoom:50%;" />

<img src=".\note_figs\image-20211027153521881.png" alt="image-20211027153521881" style="zoom:50%;" />

![image-20211027153820572](.\note_figs\image-20211027153820572.png)

## Sequence Labeling

<img src=".\note_figs\image-20211027154521355.png" alt="image-20211027154521355" style="zoom:50%;" />

<img src=".\note_figs\image-20211027154646336.png" alt="image-20211027154646336" style="zoom:50%;" />

<img src=".\note_figs\image-20211027155010725.png" alt="image-20211027155010725" style="zoom:50%;" />

考虑整个序列

不只是考虑单独一个vector

 <img src=".\note_figs\image-20211027155049710.png" alt="image-20211027155049710" style="zoom:50%;" />

<img src=".\note_figs\image-20211027155246853.png" alt="image-20211027155246853" style="zoom:50%;" />

<img src=".\note_figs\image-20211027160001171.png" alt="image-20211027160001171" style="zoom:50%;" />

<img src=".\note_figs\image-20211027160343316.png" alt="image-20211027160343316" style="zoom:50%;" />

找到跟a1相关的Sequence

类似RNN

求相关性：1.点积

<img src=".\note_figs\image-20211027160537438.png" alt="image-20211027160537438" style="zoom:50%;" />

2. Additive

   <img src=".\note_figs\image-20211027160638520.png" alt="image-20211027160638520" style="zoom:50%;" />

   主要用第一种方法，点积，具体步骤：

   <img src=".\note_figs\image-20211027160855163.png" alt="image-20211027160855163" style="zoom:50%;" />

   <img src=".\note_figs\image-20211027160925764.png" alt="image-20211027160925764" style="zoom:50%;" />

   根据这个a‘去Sequence抽取出重要的资讯。

   每一个v都乘a’，求和

   哪些求和值比较高，关联度就高

   <img src=".\note_figs\image-20211027161056913.png" alt="image-20211027161056913" style="zoom:50%;" />

   

   <img src=".\note_figs\image-20211027161948861.png" alt="image-20211027161948861"  />

   

   ![image-20211027161758773](.\note_figs\image-20211027161758773.png)

   

   ![image-20211027162136777](.\note_figs\image-20211027162136777.png)

   k是transpose的。

   ![image-20211027162243693](.\note_figs\image-20211027162243693.png)

   

   ![image-20211027162302516](.\note_figs\image-20211027162302516.png)

   

   ![image-20211027162453956](.\note_figs\image-20211027162453956.png)

   ![image-20211027163137422](.\note_figs\image-20211027163137422.png)

   

## 2. Muti-Head Self Attention

更高级的Attention机制

两个head：

<img src=".\note_figs\image-20211027165137905.png" alt="image-20211027165137905" style="zoom:50%;" />

![image-20211027165236766](.\note_figs\image-20211027165236766.png)



## 3. 位置编码Positional Encoding

在Self-Attention中没有位置信息！！！

每一个位置都有一个专属的位置e^i 

把e^1 + a^i

![image-20211027171240800](.\note_figs\image-20211027171240800.png)

![image-20211027171351611](.\note_figs\image-20211027171351611.png)

这个位置编码是hand crafted的，也就是人为设定的。

论文里面是通过sin 和 cos 产生的

位置编码可以learned from data !!!!

![image-20211027171627144](.\note_figs\image-20211027171627144.png)



## 4.Truncated Self-attention

<img src=".\note_figs\image-20211027173915965.png" alt="image-20211027173915965" style="zoom:67%;" />

一个句子的Sequence的vector很多，数据量很大！

ettention matrix 很大

很难训练

怎么办？

<img src=".\note_figs\image-20211027174110947.png" alt="image-20211027174110947" style="zoom:67%;" />

只看一部分的范围

不要看一整句话，看一句话某个位置的前后位置，上下文

范围是人为设定。

## 5. ViT

<img src=".\note_figs\image-20211027174214523.png" alt="image-20211027174214523" style="zoom:50%;" />

<img src=".\note_figs\image-20211027174431510.png" alt="image-20211027174431510" style="zoom:50%;" />

<img src=".\note_figs\image-20211027174456717.png" alt="image-20211027174456717" style="zoom:67%;" />

<img src=".\note_figs\image-20211027174602074.png" alt="image-20211027174602074" style="zoom:67%;" />

CNN会设定一个感受野

每一个Neural只会考虑感受野里面的信息

self-attention考虑整张图的所有信息

self-attention是复杂化的CNN

CNN感受野是人为设定的

self-attention感受野像是在自己学习出来的

<img src=".\note_figs\image-20211027174824604.png" alt="image-20211027174824604" style="zoom:67%;" />

<img src=".\note_figs\image-20211027175056685.png" alt="image-20211027175056685" style="zoom:67%;" />

想象一个pic是一个16x16word

随着数据量越来越大，selfAttention结果会比CNN好（蓝线）

但是数据量稍微少一点的时候CNN会达到比较好的结果。

SA鲁棒性大

CNN鲁棒性小

与RNN的对比：

<img src=".\note_figs\image-20211027191450978.png" alt="image-20211027191450978" style="zoom:67%;" />

并行性

速度性



GNN：

<img src=".\note_figs\image-20211027191838595.png" alt="image-20211027191838595" style="zoom:67%;" />

<img src=".\note_figs\image-20211027192002833.png" alt="image-20211027192002833" style="zoom:67%;" />

## 6.Transformer

Seq2Seq

 编码器：

<img src=".\note_figs\image-20211027194541719.png" alt="image-20211027194541719" style="zoom:67%;" />



<img src=".\note_figs\image-20211027194719180.png" alt="image-20211027194719180" style="zoom:67%;" />



<img src=".\note_figs\image-20211027195042704.png" alt="image-20211027195042704" style="zoom:67%;" />

这里使用的Normalization不是Batch Normalization， 

而是Layer Norm，输入一个向量输出一个向量。

这里的Residual是残差计算。



为什么BatchNormalization不如Layer Normalization？？

<img src=".\note_figs\image-20211027195348360.png" alt="image-20211027195348360" style="zoom:67%;" />



解码器： 

<img src=".\note_figs\image-20211027202752953.png" alt="image-20211027202752953" style="zoom:67%;" />

结构有点像RNN

<img src=".\note_figs\image-20211027202926221.png" alt="image-20211027202926221" style="zoom:67%;" />

<img src=".\note_figs\image-20211027203029774.png" alt="image-20211027203029774" style="zoom:67%;" />

编码器与解码器差异：

<img src=".\note_figs\image-20211027203054084.png" alt="image-20211027203054084" style="zoom:67%;" />

除去红框部分

编码器与解码器都差不多

还有masked (防止一步错步步错？)



**原来的Self-Attention：**

<img src=".\note_figs\image-20211027203327793.png" alt="image-20211027203327793" style="zoom:67%;" />

**Mask Self-Attention：**

<img src=".\note_figs\image-20211027203402309.png" alt="image-20211027203402309" style="zoom:67%;" />

<img src=".\note_figs\image-20211027203516660.png" alt="image-20211027203516660" style="zoom:67%;" />

encoder里面的Self-Attention是考虑整个Sequence

但是decoder的输出是一个个进行计算，而不是整个Sequence进行计算。

**停止符号**

<img src=".\note_figs\image-20211027203948095.png" alt="image-20211027203948095" style="zoom:67%;" />



## 7. AT VS NAT

<img src=".\note_figs\image-20211028100815707.png" alt="image-20211028100815707" style="zoom: 50%;" />



## 8. Back to Transformer

<img src=".\note_figs\image-20211028101307077.png" alt="image-20211028101307077" style="zoom: 50%;" />

<img src=".\note_figs\image-20211028101447463.png" alt="image-20211028101447463" style="zoom: 33%;" />

 

## 9. Training

<img src=".\note_figs\image-20211028110523690.png" alt="image-20211028110523690" style="zoom:50%;" />

softmax之后输出的第一个字跟“机”越接近越好

机one hot 之后有一个特定的编码

用crossEntropy积算GT和Distribution，希望这个值很小。

跟分类有点相似。

<img src=".\note_figs\image-20211028110834833.png" alt="image-20211028110834833" style="zoom: 40%;" />

<img src=".\note_figs\image-20211028110944336.png" alt="image-20211028110944336" style="zoom:50%;" />

## 10.Copy Mechanism

不一定所有训练来源都来自输入数据

可以从输入数据复制一些出来

<img src=".\note_figs\image-20211028111239836.png" alt="image-20211028111239836" style="zoom:50%;" />



<img src=".\note_figs\image-20211028111258943.png" alt="image-20211028111258943" style="zoom:50%;" />

## 11. Guided Attention

关键词：

- Monotonic Attention
- Location-aware attention

人为引导进行Attention。

在语音识别中：

<img src=".\note_figs\image-20211028111923748.png" alt="image-20211028111923748" style="zoom:50%;" />

先看最左边词汇产生声音，然后看右边。

但是机器的Attention机制是颠三倒四的。

<img src=".\note_figs\image-20211028112009020.png" alt="image-20211028112009020" style="zoom:50%;" />

### Beam Search

<img src=".\note_figs\image-20211028112344690.png" alt="image-20211028112344690" style="zoom:50%;" />

红色是贪心解码器。

绿色是最佳路线。

但是不可能搜索所有路径。

### 优化

<img src=".\note_figs\image-20211028113137760.png" alt="image-20211028113137760" style="zoom:50%;" />

![image-20211028113247846](.\note_figs\image-20211028113247846.png)

### Exposure Bias

一步错，步步错

<img src=".\note_figs\image-20211028113415759.png" alt="image-20211028113415759" style="zoom:40%;" />

给Decoder给一些错误的数据







