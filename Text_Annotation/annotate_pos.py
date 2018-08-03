import tensorflow as tf
from . import model_crf, model_softmax
import pickle
import os
from jieba.posseg import lcut

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def annotate_pos(num_units=128,
                 num_layers=2,
                 num_tags=5,
                 model='crf',
                 model_path=None,
                 data_process_path=None):
    with open(data_process_path, mode='rb') as f:
        data_process = pickle.load(f)

    num_words = data_process.num_words
    word_index = data_process.word_index

    while True:
        print('\n使用前请确保有模型。输入文本，quit=离开；\n请输入命令：')
        text = input()
        try:
            if text == 'quit':
                print('\n再见！')
                break

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
            y_predict = sess.run(tensors['prediction'], feed_dict={input_data: texts_seq})[0]
            y = ''
            # 逐个字标注，<名词>，[动词]
            for n, word in enumerate(text):
                if y_predict[n] != 0:
                    if y_predict[n] == 1:
                        y += (' [' + word + ',n] ')
                    elif y_predict[n] == 2:
                        y += (' [' + word)
                    elif y_predict[n] in [3, 9]:
                        y += word
                    elif y_predict[n] == 4:
                        y += (word + ',n] ')
                    elif y_predict[n] == 5:
                        y += (' [' + word + ',v] ')
                    elif y_predict[n] == 6:
                        y += (' [' + word)
                    elif y_predict[n] == 7:
                        y += word
                    elif y_predict[n] == 8:
                        y += (word + ',v] ')

            labels = ''
            for i in lcut(text):
                if list(i)[1] in ['n', 'v']:
                    labels += ' [' + list(i)[0] + ',' + list(i)[1] + '] '
                else:
                    labels += list(i)[0]

            print('\n分析结果：\n',
                  # y_predict, '\n',
                  y,
                  '\n\n正确结果：\n',
                  labels
                  )

        except Exception as e:
            print(e)
