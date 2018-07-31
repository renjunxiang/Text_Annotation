import tensorflow as tf
from . import model_clf
import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def annotate_cut(num_units=128,
                 num_layers=2,
                 num_tags=5,
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

            texts_seq = data_process.text2seq(texts=[text],
                                              num_words=num_words,
                                              word_index=word_index)

            tf.reset_default_graph()
            input_data = tf.placeholder(tf.int32, [None, None])
            output_targets = tf.placeholder(tf.int32, [None, None])

            tensors = model_clf(input_data=input_data,
                                output_targets=output_targets,
                                num_words=num_words,
                                num_units=num_units,
                                num_layers=num_layers,
                                batchsize=1,
                                num_tags=num_tags,
                                max_seq_len=len(text))

            saver = tf.train.Saver(tf.global_variables())
            initializer = tf.global_variables_initializer()

            sess = tf.Session()
            sess.run(initializer)
            checkpoint = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, checkpoint)
            y_predict = sess.run(tensors['prediction'], feed_dict={input_data: texts_seq})[0]
            y = ''
            for n, word in enumerate(text):
                if y_predict[n] != 0:
                    y += word
                    if y_predict[n] in [1, 4]:
                        y += '/'
            print('\n分析结果：\n',
                  # y_predict, '\n',
                  y)

        except Exception as e:
            print(e)
