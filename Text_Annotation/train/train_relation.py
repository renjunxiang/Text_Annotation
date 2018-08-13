from ..net import SklearnClf, DL
from sklearn.externals import joblib


def train_relation(x=None,
                   y=None,
                   num_tag=None,
                   method='SVM',
                   batchsize=64,
                   epoch=1,
                   model_path=None):
    """
    关系分类器
    :param x: 特征变量,array
        np.array([[1, 2, 3, 4],
                  [2, 3, 4, 5],
                  [3, 4, 5, 6],
                  [4, 5, 6, 7]])
    :param y: 标签,array
        np.array([[1, 0, 1, 2]])
    :param num_tag: 标签数量,int
    :param method: 方法名称,str
        SVM,Logistic,DL
    :param batchsize: int
    :param epoch: int
    :param model_path: 模型保存路径,str
    :return:
    """
    if method in ['SVM', 'Logistic']:
        model = SklearnClf(method=method)
        model.fit(X=x, y=y)
        joblib.dump(model, model_path)
    elif method == 'DL':
        model = DL(input_shape=[len(x[0])],
                   output_shape=num_tag)
        model.fit(x=x, y=y,
                  batch_size=batchsize, epochs=epoch,
                  verbose=1)
        model.save(model_path)
    else:
        raise ValueError('method should be SVM, Logistic, DL')

    return model
