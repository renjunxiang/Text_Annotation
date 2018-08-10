from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.models import Model
from keras.layers import Input, Dense, Activation, BatchNormalization


def SklearnClf(method='SVM', **param):
    '''
    Good performance in small dataset
    :param method: sklearn model name
    :param param: sklearn model name param,such as "C","kernel"...
    :return: sklearn model
    '''
    if method == 'SVM':
        model = SVC(probability=True,
                    **param)
    elif method == 'Logistic':
        model = LogisticRegression(**param)

    return model


def DL(input_shape, output_shape):
    '''
    Simple net.

    :param input_shape: Shape of the input data
    :param vec_size:Dimension of the dense embedding
    :param output_shape:Target shape,target should be int
    :return:keras model
    '''
    data_input = Input(shape=input_shape)
    x = Dense(500)(data_input)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Dense(output_shape)(x)
    data_output = Activation(activation='softmax')(x)

    model = Model(inputs=data_input, outputs=data_output)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



