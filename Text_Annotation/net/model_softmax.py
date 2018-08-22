import tensorflow as tf


def model_softmax(input_data=None,
                  output_targets=None,
                  num_words=3000,
                  num_units=128,
                  num_layers=2,
                  num_tags=5,
                  batchsize=1,
                  train=True
                  ):
    """

    :param input_data:
    :param output_targets:
    :param num_words:
    :param num_units:
    :param num_layers:
    :param num_tags:标签数量
    :param batchsize:
    :param train: 训练还是预测
    :return:
    """
    tensors = {}

    with tf.name_scope('embedding'):
        w = tf.Variable(tf.random_uniform([num_words, num_units], -1.0, 1.0), name="W")
        # 词向量shape [?,?,num_units]
        inputs = tf.nn.embedding_lookup(w, input_data)

    with tf.name_scope('lstm'):
        lstmcell = tf.nn.rnn_cell.BasicLSTMCell
        cell_list = [lstmcell(num_units, state_is_tuple=True) for i in range(num_layers)]
        cell_mul = tf.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)

        initial_state = cell_mul.zero_state(batch_size=batchsize, dtype=tf.float32)
        # 序列输出shape [?,?,num_units]
        outputs, last_state = tf.nn.dynamic_rnn(cell_mul, inputs, initial_state=initial_state)

    with tf.name_scope('softmax'):
        output = tf.reshape(outputs, [-1, num_units])
        weights = tf.Variable(tf.truncated_normal([num_units, num_tags]))
        bias = tf.Variable(tf.zeros(shape=[num_tags]))
        logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
        prediction = tf.reshape(tf.argmax(logits, axis=1, output_type=tf.int32),
                                shape=[batchsize, -1])

    # 训练的时候计算loss,target用独热编码;生成的时候只需要计算logits
    if train:
        with tf.name_scope('loss'):
            labels = tf.reshape(output_targets, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            total_loss = tf.reduce_mean(loss)
            accu = tf.reduce_mean(tf.cast(tf.equal(output_targets, prediction),
                                          dtype=tf.float32))

        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        tensors['initial_state'] = initial_state
        tensors['output'] = output
        tensors['last_state'] = last_state
        tensors['train_op'] = train_op
        tensors['prediction'] = prediction
        tensors['loss'] = total_loss
        tensors['accu'] = accu
    else:
        # 和CRF的输出保持统一
        tensors['prediction'] = prediction

    return tensors
