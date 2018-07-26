# Text_Annotation
[![](https://img.shields.io/badge/Python-3.5,3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/numpy-1.14.3-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.14.3)
[![](https://img.shields.io/badge/keras-2.1.6-brightgreen.svg)](https://pypi.python.org/pypi/keras/2.1.6)
[![](https://img.shields.io/badge/tensorflow-1.4,1.6-brightgreen.svg)](https://pypi.python.org/pypi/tensorflow/1.6.0)<br>

文本标注

## **项目介绍**
上次研究了一下文本生成，刚好工作上有实体识别的业务需求，就研究了一下文本标注。<br>
特征抽取部分的代码和文本生成没太大区别，LSTM改成了双向，最后一层的逐帧Softmax改成CRF而已。<br>
参考了几篇博文的分析：《CRF序列标注模型几个问题的理解》、《使用RNN解决NLP中序列标注问题的通用优化思路》，在此表示感谢！

## **模块简介**
### 模块结构
结构很简单，方法在Text_Annotation文件夹内，还有一个简单的demo。Text_Annotation文件夹包括：<br>
* **数据**：小黄鸡聊天记录，我自己上传了部分，完整版来源<https://github.com/fateleak/dgk_lost_conv>，在此表示感谢！<br>
* **预处理**：Data_process.py是个方法，用于载入数据、分词、编码、填充<br>
* **网络**：model_clf.py，2层双向LSTM+CRF<br>
* **训练**：train.py<br>
* **生成**：annotate.py<br>

### 一些说明
1.不同于LSTM文本生成，CRF要求数据reshape回[batchsize, max_seq_len, num_tags]，所以生成的时候务必保证网络的shape和输入文本的shape一致。<br>
<br>
2.由于没有标签数据，我用jieba对小黄鸡语料库先分词，转成BMES标签后训练，所以效果也就凑合看看啦。<br>
<br>
3.编码过程使用了keras的Tokenizer，他的num_words是包含0的，也就是保留春词语数量实际是num_words-1。不同于分类和生成，标注要注意数据不能在Tokenizer的时候删掉低频词。所以我并没有使用texts_to_sequences，而是手动把超过num_words的编码转为num_words。<br>
<br>
4.分词的话相对标注简单一些，只要BMES基本就够了，实体识别的话标注会复杂一些，每个实体类别都有BME。<br>
<br>
5.有时间我会尝试一下逐帧softmax的效果，理论上每个词的输出是结合了上下文语义，而且从唐诗生成的效果看是完全可以学到上下文的。

## 成果展示
**先要train.py进行训练，annotate.py用于预测(分词)，参考demo.py**<br>
<br>
![](https://github.com/renjunxiang/Text_Annotation/blob/master/picture/demo.jpg)<br><br>