from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import os
import pickle
import re
import jieba

jieba.setLogLevel('WARN')

DIR = os.path.dirname(os.path.abspath(__file__))


class Data_process():
    def __init__(self):
        self.texts = None
        self.len_max = None
        self.word_index = None
        self.num_words = None
        self.texts_seq = None
        self.maxlen = None

    def load_data(self, len_min=0, len_max=200, num=50000):
        '''
        导入数据，长度控制适用于text2seq的mode='poem'
        :param file: 数据源，poem诗歌，story小说
        :param len_min: 最短长度
        :param len_max: 最长长度
        :return:
        '''
        identifier = {1: 'S', 2: 'B', 3: 'M', 4: 'E'}
        with open(DIR + '/data/xiaohuangji50w_nofenci.conv', encoding='utf-8', mode='r') as f:
            texts = []
            line = True
            n = 0
            while line:
                line = f.readline()
                if line != 'E\n':
                    line_sub = re.sub(pattern='[M\s]', repl='', string=line)
                    if len(line_sub) > len_min and len(line_sub) < len_max:
                        texts.append(line_sub)
                        n += 1
                        if n >= num:
                            break
            f.close()

        target = []
        for text in texts:
            text_cut = jieba.lcut(text)
            text_target = []
            for word in text_cut:
                # 单个字标识为S
                if len(word) == 1:
                    text_target += [1]
                # 词语标识为BMS
                else:
                    text_target += ([2] + [3] * (len(word) - 2) + [4])
            target.append(text_target)

        self.len_min = len_min
        self.len_max = len_max
        self.identifier = identifier
        return texts, target

    def pad(self, seq, max_len, pad_value=0):
        seq_pad = np.pad(seq,
                         pad_width=(0, max_len - len(seq)),
                         mode='constant',
                         constant_values=pad_value)
        return seq_pad

    def text2seq(self, texts=None, num_words=5000, word_index=None):
        '''
        文本转编码
        :param mode: 文本reshape方式，length以长度重新分割，sample按样本
        :param num_words: 保留词语数量
        :param maxlen: 保留文本长度
        :return:
        '''
        if word_index is None:
            tokenizer = Tokenizer(num_words=num_words, char_level=True)
            # 训练词典,实际保留的是num_words-1个词,还有一个是0
            tokenizer.fit_on_texts(texts)
            word_index = tokenizer.word_index
        num_words = min(num_words, len(word_index.keys()) + 1)
        # words = sorted(word_index.keys(), key=lambda x: word_index[x])[:num_words - 1]
        # word_index = {words[i - 1]: i for i in range(1, num_words)}
        # index_word = {i: words[i - 1] for i in range(1, num_words)}
        # 超过num_words的定义为U
        # index_word.update({num_words: 'U'})
        # self.word_index = word_index
        # self.index_word = index_word
        self.num_words = num_words
        self.word_index = word_index

        # 转编码
        texts_seq = []
        for text in texts:
            text_seq = []
            for word in text:
                # 超过num_words-1的编码为num_words
                if word_index[word] < num_words:
                    text_seq.append(word_index[word])
                else:
                    text_seq.append(num_words)
            texts_seq.append(text_seq)

        return texts_seq

    def data_transform(self, len_min=0, len_max=200,
                       num_words=5000, num=50000):
        # 导入数据
        texts, target = self.load_data(len_min=len_min, len_max=len_max, num=50000)
        # 转编码
        texts_seq = self.text2seq(texts=texts, num_words=num_words)
        max_seq_len = np.array([len(i) for i in texts_seq]).max()
        # 补齐长度
        texts_seq = np.array([self.pad(i, max_seq_len, 0) for i in texts_seq])
        target = np.array([self.pad(i, max_seq_len, 0) for i in target])
        self.max_seq_len = max_seq_len

        return texts_seq, target
