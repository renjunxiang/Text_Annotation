from Text_Annotation import Data_process
from Text_Annotation.train import train_annotation
from Text_Annotation.demo import annotate_pos
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
texts_seq, target = data_process.data_transform(len_min=0,
                                                len_max=200,
                                                num_words=5000,
                                                num=100000,
                                                file='knowledge')

with open(DIR + '/model/%s/model_pos/data_process.pkl' % (params['model']), mode='wb') as f:
    pickle.dump(data_process, f)

train_annotation(x=texts_seq,
                 y=target,
                 num_words=data_process.num_words,
                 batchsize=64,
                 epoch=10,
                 max_seq_len=data_process.max_seq_len,
                 model_path=DIR + '/model/%s/model_pos/' % (params['model']),
                 **params)

annotate_pos(data_process_path=DIR + '/model/%s/model_pos/data_process.pkl' % (params['model']),
             model_path=DIR + '/model/%s/model_pos/' % (params['model']),
             **params)
