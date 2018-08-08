import tensorflow as tf
from ..net import model_crf, model_softmax
import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def annotate(text=None,
             num_units=128,
             num_layers=2,
             num_tags=5,
             model='crf',
             model_path=None,
             data_process_path=None):
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

    sess = tf.Session()
    sess.run(initializer)

    # 导入模型系数
    checkpoint = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, checkpoint)
    y_predict, output_fb = sess.run([tensors['prediction'], tensors['output_fb']],
                                    feed_dict={input_data: texts_seq})[0]

    return y_predict, output_fb
