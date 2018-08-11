import numpy as np
from Text_Annotation.data import creat_relation
from Text_Annotation import Data_process
from Text_Annotation.annotate import annotate
import os

DIR = os.path.dirname(os.path.abspath(__file__))
# DIR = '.'

params = {
    'model': 'crf',
    'num_units': 128,
    'num_layers': 2,
    'num_tags': 10,
}

data_process = Data_process()
texts, targets = data_process.load_data(file='knowledge')

regulations = [['n', [1]],
               ['n', [2, 3, 4]],
               ['v', [5]],
               ['v', [6, 7, 8]],
               ['U', [9]]]

train_x, train_y = [], []
for num, text in enumerate(texts):
    output_fb = annotate(text=text,
                         data_process_path=DIR + '/model/%s/model_pos/data_process.pkl' % (params['model']),
                         model_path=DIR + '/model/%s/model_pos/' % (params['model']),
                         train=True,
                         **params)

    train_x_one, train_y_one = creat_relation(output_fb[0],
                                              regulations,
                                              targets[num])
    train_x += train_x_one
    train_y += train_y_one
    print(num)

np.save(DIR + '/data/train_x.npy', train_x)
np.save(DIR + '/data/train_y.npy', train_y)
