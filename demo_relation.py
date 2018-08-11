from Text_Annotation.annotate import annotate, find_relation
from Text_Annotation.train import train_relation
from keras.models import load_model
import numpy as np
import os

DIR = os.path.dirname(os.path.abspath(__file__))

params = {
    'model': 'crf',
    'num_units': 128,
    'num_layers': 2,
    'num_tags': 10,
}

# train_x = np.load(DIR + '/data/train_x.npy')
# train_y = np.load(DIR + '/data/train_y.npy')
#
# # model=train_relation(x=train_x,y=train_y,method='SVM',model_path=DIR+'/model/relation/SVM.model')
# model = train_relation(x=train_x, y=train_y, num_tag=3,
#                        batchsize=64, epoch=1,
#                        method='DL', model_path=DIR + '/model/relation/DL.h5')

method = 'DL'
model = load_model(DIR + '/model/relation/DL.h5')
# 不先使用一下keras的模型后续会报计算图错误...
m = model.predict(np.ones([1, 512]))

regulations = [['n', [1]],
               ['n', [2, 3, 4]],
               ['v', [5]],
               ['v', [6, 7, 8]],
               ['U', [9]]]
regular = [['v', 'n'], ['n', 'v']]

while True:
    print('\n使用前请确保有模型。输入文本，quit=离开；\n请输入命令：')
    text = input()

    if text == 'quit':
        print('\n再见！')
        break

    # text='医疗机构变更单位名称、法定代表人或负责人'
    y_predict, output_fb = annotate(text=text,
                                    data_process_path=DIR + '/model/%s/model_pos/data_process.pkl' % (params['model']),
                                    model_path=DIR + '/model/%s/model_pos/' % (params['model']),
                                    train=False,
                                    **params)

    result = find_relation(text=text,
                           sentence_vector=output_fb[0],
                           regulations=regulations,
                           regular=regular,
                           annotation=y_predict[0],
                           model=model,
                           method='DL',
                           tags=['null', '主谓', '动宾'])

    print('\n实体识别：\n')
    for i in result['entities']:
        print(i)
    print('\n关系抽取：\n')
    for i in result['relations']:
        print('实体1',i['entity1'])
        print('实体2', i['entity2'])
        print('关系', i['relation'])
        print('\n')
    # print('\n分析结果：\n',
    #       result)
