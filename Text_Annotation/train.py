import numpy as np
import tensorflow as tf
from . import model_clf

def train(x=None,
          y=None,
          num_words=5000,
          num_units=128,
          num_layers=2,
          num_tags=5,
          max_seq_len=20,
          batchsize=64,
          epoch=1):
    input_data = tf.placeholder(tf.int32, [None, None])
    output_targets = tf.placeholder(tf.int32, [None, None])

    tensors = model_clf(input_data=input_data,
                        output_targets=output_targets,
                        num_words=num_words,
                        num_units=num_units,
                        num_layers=num_layers,
                        batchsize=batchsize,
                        num_tags=num_tags,
                        max_seq_len=max_seq_len)

    saver = tf.train.Saver(tf.global_variables())
    initializer = tf.global_variables_initializer()
    print('start training')

    with tf.Session() as sess:
        sess.run(initializer)
        index_all = np.arange(len(x))

        for epoch in range(epoch):
            for batch in range(len(x) // batchsize * 1):
                index_batch = np.random.choice(index_all, batchsize)

                x_batch = x[index_batch]
                y_batch = y[index_batch]

                print(x_batch.shape)
                loss, _ = sess.run([
                    tensors['loss'],
                    tensors['train_op']
                ], feed_dict={input_data: x_batch, output_targets: y_batch})
                print('Epoch: %d, batch: %d, loss loss: %.6f' % (epoch + 1, batch + 1, loss))
            saver.save(sess, './model/', global_step=epoch)
