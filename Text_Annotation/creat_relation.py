import numpy as np
from .annotate import locate, pair_vector
from Text_Annotation import Data_process
from Text_Annotation.annotate import annotate
import pickle
import os

DIR = os.path.dirname(os.path.abspath(__file__))
params = {
    'model': 'crf',
    'num_units': 128,
    'num_layers': 2,
    'num_tags': 10,
}

data_process = Data_process()
texts, target = data_process.load_data(file='knowledge')

regulations = [['n', [1]],
               ['n', [2, 3, 4]],
               ['v', [5]],
               ['v', [6, 7, 8]],
               ['U', [9]]]


def creat_relation(sentence_vector,
                   regulations,
                   annotation):
    locations = locate(regulations, annotation)

    regular = [['v', 'n'], ['n', 'v']]
    vector_pairs = pair_vector(sentence_vector, locations, regular)

    train_x = []
    train_y = []
    for num, location in enumerate(locations):
        train_x.append(vector_pairs[num][0])
        # 动名词是否相邻,相邻标记为1,否则为0
        if location[0][0] - location[0][-1] == 1:
            train_y.append([0.0, 1.0])
        else:
            train_y.append([1.0, 0.0])

    return train_x, train_y


y_predict, output_fb = annotate(text=texts,
                                data_process_path=DIR + '/%s/model_pos/data_process.pkl' % (params['model']),
                                model_path=DIR + '/%s/model_pos/' % (params['model']),
                                **params)
