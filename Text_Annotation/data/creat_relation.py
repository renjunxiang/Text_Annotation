import numpy as np
from ..annotate import locate, pair_vector, annotate
import os
import time

DIR = os.path.dirname(os.path.abspath(__file__))


def creat_relation(sentence_vector,
                   regulations,
                   regular,
                   annotation):
    """
    标注结果 12341234 -> 实体定位 [[[0, 1], 'v'], [[2, 3, 4, 5], 'n']]
    CRF模型 -> 句子向量 [[6, 6], [8, 8], [2, 2], [3, 3]]
    句子向量+实体定位 - > 实体组合 [[[7, 7, 3, 3], ['v', 'n']], [[3, 3, 7, 7], ['n', 'v']]]
    实体组合 -> 训练集标签 [[[7, 7, 3, 3], [0, 1]], [[3, 3, 7, 7], [1, 0]]]
    :param sentence_vector:句子向量
    :param regulations:标注规则
    :param regular:配对规则
    :param annotation:标注结果
    :return:
    """
    locations = locate(regulations, annotation)
    entity_pairs, vector_pairs = pair_vector(sentence_vector, locations, regular)

    train_x = []
    train_y = []
    for num, vector_pair in enumerate(vector_pairs):
        train_x.append(vector_pair[0])
        # 动名词是否相邻,主谓标记为1,动宾标记为2,,否则为0
        if entity_pairs[num][1][0][0] - entity_pairs[num][0][0][-1] == 1:
            if [entity_pairs[num][0][1], entity_pairs[num][1][1]] == ['n', 'v']:
                train_y.append([1])
            elif [entity_pairs[num][0][1], entity_pairs[num][1][1]] == ['v', 'n']:
                train_y.append([2])
        else:
            train_y.append([0])

    return train_x, train_y


def creat_relations(texts,
                    targets,
                    data_process_path,
                    model_path,
                    x_path,
                    y_path,
                    lock=None):
    params = {
        'model': 'crf',
        'num_units': 128,
        'num_layers': 2,
        'num_tags': 10,
    }

    regulations = [['n', [1]],
                   ['n', [2, 3, 4]],
                   ['v', [5]],
                   ['v', [6, 7, 8]],
                   ['U', [9]]]
    regular = [['v', 'n'], ['n', 'v']]

    train_x, train_y = [], []
    for num, text in enumerate(texts):
        print(num)
        if lock is not None:
            lock.acquire()
        output_fb = annotate(text=text,
                             data_process_path=data_process_path,
                             model_path=model_path,
                             train=True,
                             **params)
        if lock is not None:
            lock.release()
        time.sleep(0.05)

        train_x_one, train_y_one = creat_relation(sentence_vector=output_fb[0],
                                                  regulations=regulations,
                                                  regular=regular,
                                                  annotation=targets[num])
        train_x += train_x_one
        train_y += train_y_one

    np.save(x_path, train_x)
    np.save(y_path, train_y)
