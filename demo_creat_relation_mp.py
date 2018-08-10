import numpy as np
from Text_Annotation import Data_process
from Text_Annotation.data import creat_relations
from multiprocessing import Process, Lock
import os

if __name__ == '__main__':
    # DIR = os.path.dirname(os.path.abspath(__file__))
    DIR = '.'

    params = {
        'model': 'crf',
        'num_units': 128,
        'num_layers': 2,
        'num_tags': 10,
    }
    data_process = Data_process()
    texts, targets = data_process.load_data(file='knowledge')

    process_list = []
    process_num = 10
    sample = 20
    lock = Lock()


    for i in range(process_num):
        texts_p = texts[(i * sample):((i + 1) * sample)]
        targets_p = targets[(i * sample):((i + 1) * sample)]
        po = Process(target=creat_relations,
                     kwargs={'texts': texts_p,
                             'targets': targets_p,
                             'data_process_path': DIR + '/%s/model_pos/data_process.pkl' % (params['model']),
                             'model_path': DIR + '/%s/model_pos/' % (params['model']),
                             'x_path': DIR + '/data/train_x_%d.npy' % i,
                             'y_path': DIR + '/data/train_y_%d.npy' % i,
                             'lock':lock
                             })
        process_list.append(po)

    # 启动子进程
    for process in process_list:
        process.start()
    # 等待子进程全部结束
    for process in process_list:
        process.join()

    train_x = []
    train_y = []
    for i in range(process_num):
        train_x_one = np.load(DIR + '/data/train_x_%d.npy' % i)
        train_y_one = np.load(DIR + '/data/train_y_%d.npy' % i)
        train_x += train_x_one.tolist()
        train_y += train_y_one.tolist()
    np.save(DIR + '/data/train_x.npy', np.array(train_x, dtype=np.float32))
    np.save(DIR + '/data/train_y.npy', np.array(train_y, dtype=np.float32))

# import numpy as np
# train_x_one = np.load('./data/train_x_%d.npy' % 0)
# train_x = np.load('./data/train_x.npy')
# np.save(DIR + '/data/train_x1.npy', np.array(train_x,dtype=np.float32))
# len(train_x_one)
# len(train_x)
#
# train_x_one[0]
# train_x[0]


# print(train_x_one.shape)
#
# x=np.array([])
# y=np.array([[1,2,3],[4,5,6]])
#
# np.append(x,y)
# np.concatenate([x,y],axis=1)
