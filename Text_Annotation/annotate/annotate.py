import tensorflow as tf
from ..net import model_crf, model_softmax
from .location import locate, pair_vector
from collections import OrderedDict
import pickle
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def annotate(text=None,
             num_units=128,
             num_layers=2,
             num_tags=5,
             model='crf',
             model_path=None,
             data_process_path=None,
             train=True):
    with open(data_process_path, mode='rb') as f:
        data_process = pickle.load(f)

    num_words = data_process.num_words
    word_index = data_process.word_index

    # 文本转编码
    texts_seq = data_process.text2seq(texts=[text],
                                      num_words=num_words,
                                      word_index=word_index)

    tf.reset_default_graph()
    input_data = tf.placeholder(tf.int32, [None, None])
    output_targets = tf.placeholder(tf.int32, [None, None])

    # 导入计算图
    if model == 'crf':
        tensors = model_crf(input_data=input_data,
                            output_targets=output_targets,
                            num_words=num_words,
                            num_units=num_units,
                            num_layers=num_layers,
                            batchsize=1,
                            num_tags=num_tags,
                            max_seq_len=len(text))
    elif model == 'softmax':
        tensors = model_softmax(input_data=input_data,
                                output_targets=output_targets,
                                num_words=num_words,
                                num_units=num_units,
                                num_layers=num_layers,
                                batchsize=1,
                                num_tags=num_tags)

    saver = tf.train.Saver(tf.global_variables())
    initializer = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(initializer)

        # 导入模型系数
        checkpoint = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, checkpoint)
        if train:
            output_fb = sess.run(tensors['output_fb'],
                                 feed_dict={input_data: texts_seq})
            return output_fb
        else:
            y_predict, output_fb = sess.run([tensors['prediction'], tensors['output_fb']],
                                            feed_dict={input_data: texts_seq})
            return y_predict, output_fb


def find_relation(text,
                  sentence_vector,
                  regulations,
                  regular,
                  annotation,
                  model,
                  method='DL',
                  tags=None):
    """
    标注结果 12341234 -> 实体定位 [[[0, 1], 'v'], [[2, 3, 4, 5], 'n']]
    CRF模型 -> 句子向量 [[6, 6], [8, 8], [2, 2], [3, 3]]
    句子向量+实体定位 - > 实体组合 [[[7, 7, 3, 3], ['v', 'n']], [[3, 3, 7, 7], ['n', 'v']]]
    实体组合 -> 预测关系标签 [[[7, 7, 3, 3], [0, 1]], [[3, 3, 7, 7], [1, 0]]]
    :param text:文本
    :param sentence_vector:句子向量
    :param regulations:标注规则
    :param regular:配对规则[['v', 'n'], ['n', 'v']]
    :param annotation:标注结果
    :param model:模型
    :param method:模型名称,SVM,Logistic,DL
    :param tags:关系标签,list,['null','nv']
    :return:

    result = {
    'text': '我喜欢吃苹果',
    'entity': [
        {
            'text': '我',
            'location': [0],
            'type': 'n'
        },
        {
            'text': '喜欢',
            'location': [1, 2],
            'type': 'v'
        },
        {
            'text': '吃',
            'location': [3],
            'type': 'v'
        },
        {
            'text': '苹果',
            'location': [4, 5],
            'type': 'n'
        }
    ],
    'relation': [
        {
            'entity1': {
                'text': '我',
                'location': [0],
                'type': 'n'
            },
            'entity2': {
                'text': '喜欢',
                'location': [1, 2],
                'type': 'v'
            },
            'relation': {
                'score': 0.8,
                'classify': '主谓'
            }
        },
        {
            'entity1': {
                'text': '吃',
                'location': [3],
                'type': 'v'
            },
            'entity2': {
                'text': '苹果',
                'location': [4, 5],
                'type': 'n'
            },
            'relation': {
                'score': 0.9,
                'classify': '动宾'
            }
        }
    ]
}
    """
    # results = OrderedDict()
    results = {}
    results['text'] = text
    results['entities'] = []
    results['relations'] = []

    # 实体识别
    locations = locate(regulations, annotation)
    for location in locations:
        # result_entity = OrderedDict()
        if location[1] != 'U':
            result_entity = {}
            result_entity['text'] = ''.join([text[i] for i in location[0]])
            result_entity['location'] = location[0]
            result_entity['type'] = location[1]
            results['entities'].append(result_entity)

    # 关系抽取
    entity_pairs, vector_pairs = pair_vector(sentence_vector, locations, regular)

    for num, vector_pair in enumerate(vector_pairs):
        result_relation = {
            'entity1': {'text': '我', 'location': [0], 'type': 'n'},
            'entity2': {'text': '喜欢', 'location': [1, 2], 'type': 'v'},
            'relation': {'score': 0.8, 'classify': 'n_v'}
        }
        entity_pair = entity_pairs[num]

        result_relation['entity1']['text'] = ''.join([text[i] for i in entity_pair[0][0]])
        result_relation['entity1']['location'] = entity_pair[0][0]
        result_relation['entity1']['type'] = entity_pair[0][1]

        result_relation['entity2']['text'] = ''.join([text[i] for i in entity_pair[1][0]])
        result_relation['entity2']['location'] = entity_pair[1][0]
        result_relation['entity2']['type'] = entity_pair[1][1]

        # keras直接输出概率,sklearn需要指定predict_prob
        if method == 'DL':
            relation = model.predict(np.array([vector_pair[0]]))[0]
        else:
            relation = model.predict_prob(np.array([vector_pair[0]]))[0]

        relation_index = np.argmax(relation)
        result_relation['relation']['score'] = relation[relation_index]
        result_relation['relation']['classify'] = tags[relation_index]

        # 只保留有意义的关系
        if relation_index != 0:
            results['relations'].append(result_relation)

    return results
