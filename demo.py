from Text_Annotation import Data_process, train, annotate
import pickle
import os

DIR = os.path.dirname(os.path.abspath(__file__))
params = {
    'num_units': 128,
    'num_layers': 2,
    'num_tags': 5
}

data_process = Data_process()
texts_seq, target = data_process.data_transform(len_min=0,
                                                len_max=10,
                                                num_words=5000,
                                                num=100000)

with open('./model/data_process.pkl', mode='wb') as f:
    pickle.dump(data_process, f)

train(x=texts_seq,
      y=target,
      num_words=data_process.num_words,
      batchsize=64,
      epoch=1,
      max_seq_len=data_process.max_seq_len,
      **params)

annotate(model_path=DIR + '/model/',
         data_process_path=DIR + '/model/data_process.pkl',
         **params)

