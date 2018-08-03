from Text_Annotation import Data_process, train, annotate_cut
import pickle
import os

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = '.'
params = {
    'model': 'crf',
    'num_units': 128,
    'num_layers': 2,
    'num_tags': 5,
}

data_process = Data_process()
texts_seq, target = data_process.data_transform(len_min=0,
                                                len_max=20,
                                                num_words=5000,
                                                num=300000,
                                                file='chat')

with open(DIR + '/%s/model_cut/data_process.pkl' % (params['model']), mode='wb') as f:
    pickle.dump(data_process, f)

train(x=texts_seq,
      y=target,
      num_words=data_process.num_words,
      batchsize=64,
      epoch=1,
      max_seq_len=data_process.max_seq_len,
      model_path=DIR + '/%s/model_cut/' % (params['model']),
      **params)

annotate_cut(data_process_path=DIR + '/%s/model_cut/data_process.pkl' % (params['model']),
             model_path=DIR + '/%s/model_cut/' % (params['model']),
             **params)
