# OCR BY TRANSFORMER

```
# 如何将图片的信息构造成transformer想要的，类似于 word embedding 形式的输入?

# 待预测的图片都是长条状的，
#
# 文字基本都是水平排列，那么我们将特征图沿水平方向进行整合，
#
# 得到的每一个embedding可以认为是图片纵向的某个切片的特征，
#
# 将这样的特征序列交给transformer，利用其强大的attention能力来完成预测

# 之间的差异主要是多了借助一个CNN网络作为backbone提取图像特征
#
# 得到input embedding的过程

# 主要有以下几个部分：
#
# 构建dataset → 图像预处理、label处理等；
# 模型构建 → backbone + transformer；
# 模型训练
# 推理 → 贪心解码
```

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter06/6.2/transformer.jpg" alt="img" style="zoom:67%;" />

# Pipeline：

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter06/6.2/ocr_by_transformer.png" alt="img" style="zoom: 67%;" />

# 基本参数设定

```
base_data_dir = '../dataset/ICDAR_2015/'    # 数据集根目录，请将数据下载到此位置
device = torch.device('cuda')  # 'cpu'或者'cuda'
nrof_epochs = 100  # 迭代次数，1500，根据需求进行修正
batch_size = 32     # 批量大小，64，根据需求进行修正
model_save_path = './log/ex1_ocr_model.pth'

# 读取图像label中字符与其id的映射字典，后续Dataset创建需要使用。
# 读取label-id映射关系记录文件
lbl2id_map_path = os.path.join(base_data_dir, 'lbl2id_map.txt')
lbl2id_map, id2lbl_map = load_lbl2id_map(lbl2id_map_path)

# 统计数据集中出现的所有的label中包含字符最多的有多少字符，数据集构造gt（ground truth）信息需要用到
train_lbl_path = os.path.join(base_data_dir, 'train_gt.txt')
valid_lbl_path = os.path.join(base_data_dir, 'valid_gt.txt')
train_max_label_len = statistics_max_len_label(train_lbl_path)
valid_max_label_len = statistics_max_len_label(valid_lbl_path)
# 数据集中字符数最多的一个case作为制作的gt的sequence_len
sequence_len = max(train_max_label_len, valid_max_label_len)
```

# Dataset构建

![img](https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter06/6.2/img2feature.jpg)

```
图片基本都是水平长条状的，图像内容是水平排列的字符组成的单词。那么图片空间上同一纵向切片的位置，基本只有一个字符，因此纵向分辨率不需要很大，那么取 H_f=1即可；而横向的分辨率需要大一些，我们需要有不同的embedding来编码水平方向上不同字符的特征。

这里，我们就用最经典的resnet18网络作为backbone，由于其下采样倍数为32，最后一层特征图channel数为512，那么:
```

Hi=Hf∗32=32

C_f = 512

### 确定宽度：

![img](https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter06/6.2/two_resize.jpg)

```
方法一：设定一个固定尺寸，将图像保持其宽高比进行resize，右侧空余区域进行padding；

方法二：直接将原始图像强制resize到一个预设的固定尺寸。
```

方式一✔

### **encode_mask**

### **label处理**

### **decode_mask**

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter06/6.2/decode_mask.png" alt="img" style="zoom: 50%;" />

















