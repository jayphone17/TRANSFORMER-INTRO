# 笔记

## 1. Self-Attention

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027153126440.png" alt="image-20211027153126440" style="zoom:50%;" />

缺点：看不出关系

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027153229726.png" alt="image-20211027153229726" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027153315383.png" alt="image-20211027153315383" style="zoom: 50%;" />

一排向量

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027153407036.png" alt="image-20211027153407036" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027153521881.png" alt="image-20211027153521881" style="zoom:50%;" />

![image-20211027153820572](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027153820572.png)

## Sequence Labeling

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027154521355.png" alt="image-20211027154521355" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027154646336.png" alt="image-20211027154646336" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027155010725.png" alt="image-20211027155010725" style="zoom:50%;" />

考虑整个序列

不只是考虑单独一个vector

 <img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027155049710.png" alt="image-20211027155049710" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027155246853.png" alt="image-20211027155246853" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027160001171.png" alt="image-20211027160001171" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027160343316.png" alt="image-20211027160343316" style="zoom:50%;" />

找到跟a1相关的Sequence

类似RNN

求相关性：1.点积

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027160537438.png" alt="image-20211027160537438" style="zoom:50%;" />

2. Additive

   <img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027160638520.png" alt="image-20211027160638520" style="zoom:50%;" />

   主要用第一种方法，点积，具体步骤：

   <img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027160855163.png" alt="image-20211027160855163" style="zoom:50%;" />

   <img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027160925764.png" alt="image-20211027160925764" style="zoom:50%;" />

   根据这个a‘去Sequence抽取出重要的资讯。

   每一个v都乘a’，求和

   哪些求和值比较高，关联度就高

   <img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027161056913.png" alt="image-20211027161056913" style="zoom:50%;" />

   

   <img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027161948861.png" alt="image-20211027161948861"  />

   

   ![image-20211027161758773](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027161758773.png)

   

   ![image-20211027162136777](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027162136777.png)

   k是transpose的。

   ![image-20211027162243693](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027162243693.png)

   

   ![image-20211027162302516](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027162302516.png)

   

   ![image-20211027162453956](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027162453956.png)

   ![image-20211027163137422](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027163137422.png)

   

## 2. Muti-Head Self Attention

更高级的Attention机制

两个head：

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027165137905.png" alt="image-20211027165137905" style="zoom:50%;" />

![image-20211027165236766](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027165236766.png)



## 3. 位置编码Positional Encoding

在Self-Attention中没有位置信息！！！

每一个位置都有一个专属的位置e^i 

把e^1 + a^i

![image-20211027171240800](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027171240800.png)

![image-20211027171351611](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027171351611.png)

这个位置编码是hand crafted的，也就是人为设定的。

论文里面是通过sin 和 cos 产生的

位置编码可以learned from data !!!!

![image-20211027171627144](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027171627144.png)



## 4.Truncated Self-attention

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027173915965.png" alt="image-20211027173915965" style="zoom:67%;" />

一个句子的Sequence的vector很多，数据量很大！

ettention matrix 很大

很难训练

怎么办？

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027174110947.png" alt="image-20211027174110947" style="zoom:67%;" />

只看一部分的范围

不要看一整句话，看一句话某个位置的前后位置，上下文

范围是人为设定。

## 5. ViT

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027174214523.png" alt="image-20211027174214523" style="zoom:50%;" />

<img src="C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027174431510.png" alt="image-20211027174431510" style="zoom:50%;" />

![image-20211027174456717](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027174456717.png)

![image-20211027174602074](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027174602074.png)

CNN会设定一个感受野

每一个Neural只会考虑感受野里面的信息

self-attention考虑整张图的所有信息

self-attention是复杂化的CNN

CNN感受野是人为设定的

self-attention感受野像是在自己学习出来的

![image-20211027174824604](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027174824604.png)

![image-20211027175056685](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027175056685.png)

想象一个pic是一个16x16word

随着数据量越来越大，selfAttention结果会比CNN好（蓝线）

但是数据量稍微少一点的时候CNN会达到比较好的结果。

SA鲁棒性大

CNN鲁棒性小

与RNN的对比：

![image-20211027191450978](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027191450978.png)

并行性

速度性



GNN：

![image-20211027191838595](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027191838595.png)

![image-20211027192002833](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027192002833.png)

## 6.Transformer

Seq2Seq

 编码器：

![image-20211027194541719](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027194541719.png)



![image-20211027194719180](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027194719180.png)



![image-20211027195042704](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027195042704.png)

这里使用的Normalization不是Batch Normalization， 

而是Layer Norm，输入一个向量输出一个向量。

这里的Residual是残差计算。



为什么BatchNormalization不如Layer Normalization？？

![image-20211027195348360](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027195348360.png)



解码器： 

![image-20211027202752953](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027202752953.png)

结构有点像RNN

![image-20211027202926221](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027202926221.png)

![image-20211027203029774](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027203029774.png)

编码器与解码器差异：

![image-20211027203054084](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027203054084.png)

除去红框部分

编码器与解码器都差不多

还有masked (防止一步错步步错？)



**原来的Self-Attention：**

![image-20211027203327793](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027203327793.png)

**Mask Self-Attention：**

![image-20211027203402309](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027203402309.png)

![image-20211027203516660](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027203516660.png)

encoder里面的Self-Attention是考虑整个Sequence

但是decoder的输出是一个个进行计算，而不是整个Sequence进行计算。

**停止符号**

![image-20211027203948095](C:\Users\JayphoneLin\AppData\Roaming\Typora\typora-user-images\image-20211027203948095.png)









