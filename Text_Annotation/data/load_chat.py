import os
import re
import jieba

jieba.setLogLevel('WARN')

DIR = os.path.dirname(os.path.abspath(__file__))

# 看着玩玩，实际只用了里面的数字
identifier = {1: 'S', 2: 'B', 3: 'M', 4: 'E'}


def load_chat(len_min=0, len_max=200, num=50000):
    with open(DIR + '/xiaohuangji50w_nofenci.conv', encoding='utf-8', mode='r') as f:
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

    targets = []
    for text in texts:
        text_cut = jieba.lcut(text)
        text_target = []
        for word in text_cut:
            # 单个字标识为S
            if len(word) == 1:
                text_target += [1]
            # 词语标识为BME
            else:
                text_target += ([2] + [3] * (len(word) - 2) + [4])
        targets.append(text_target)

    return texts, targets
