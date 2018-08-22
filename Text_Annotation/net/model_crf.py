import tensorflow as tf


def model_crf(input_data=None,
              output_targets=None,
              num_words=3000,
              num_units=128,
              num_layers=2,
              batchsize=64,
              num_tags=10,
              max_seq_len=40,
              train=True):
    """

    :param input_data:
    :param output_targets:
    :param num_words:
    :param num_units:
    :param num_layers:
    :param batchsize:
    :param num_tags:标签数量
    :param max_seq_len: 句子长度
    :param train: 训练还是预测
    :return:
    """
    tensors = {}

    # 一个batch的句子长度序列
    sequence_lengths_t = tf.constant([max_seq_len] * batchsize)

    with tf.variable_scope('embedding'):
        w = tf.Variable(tf.random_uniform([num_words, num_units], -1.0, 1.0), name="w")
        # embedding=[?,?,num_units]
        embedding = tf.nn.embedding_lookup(w, input_data, name='embedding')

    with tf.variable_scope('lstm'):
        lstmcell = tf.nn.rnn_cell.BasicLSTMCell

        cell_list_f = [lstmcell(num_units, state_is_tuple=True) for i in range(num_layers)]
        cell_mul_f = tf.nn.rnn_cell.MultiRNNCell(cell_list_f, state_is_tuple=True)
        initial_state_f = cell_mul_f.zero_state(batch_size=batchsize, dtype=tf.float32)

        cell_list_b = [lstmcell(num_units, state_is_tuple=True) for i in range(num_layers)]
        cell_mul_b = tf.nn.rnn_cell.MultiRNNCell(cell_list_b, state_is_tuple=True)
        initial_state_b = cell_mul_b.zero_state(batch_size=batchsize, dtype=tf.float32)

        # outputs=[?,?,num_units]
        outputs, last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_mul_f,
                                                              cell_bw=cell_mul_b,
                                                              inputs=embedding,
                                                              initial_state_fw=initial_state_f,
                                                              initial_state_bw=initial_state_b)

    with tf.variable_scope('logits'):
        output_fw, output_bw = outputs
        output_fb = tf.concat([output_fw, output_bw], axis=-1, name='output_fb')
        # output=[?,num_units * 2]
        output = tf.reshape(output_fb, [-1, num_units * 2], name='output')
        weights = tf.get_variable(shape=[2 * num_units, num_tags], name="W")

        # 文本生成的时候用 x*W+b,不知道这里为什么不需要b?
        # bias = tf.Variable(tf.zeros(shape=[num_words]))
        # logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
        logits = tf.matmul(output, weights)

    with tf.variable_scope('crf'):
        # unary_scores [batchsize, max_seq_len, num_tags]
        unary_scores = tf.reshape(logits, [batchsize, max_seq_len, num_tags])

        # log_likelihood极大似然估计,[batchsize];transition_params概率转移矩阵[num_tags,num_tags]
        crf_log_likelihood = tf.contrib.crf.crf_log_likelihood(inputs=unary_scores,
                                                               tag_indices=output_targets,
                                                               sequence_lengths=sequence_lengths_t)

        log_likelihood, transition_params = crf_log_likelihood
        decode_tags, best_score = tf.contrib.crf.crf_decode(potentials=unary_scores,
                                                            transition_params=transition_params,
                                                            sequence_length=sequence_lengths_t)

        loss = tf.reduce_mean(-log_likelihood)
        accu=tf.reduce_mean(tf.cast(tf.equal(output_targets,decode_tags),
                                    dtype=tf.float32))

    if train:
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        tensors['output_fb'] = output_fb
        tensors['prediction'] = decode_tags
        tensors['train_op'] = train_op
        tensors['loss'] = loss
        tensors['accu'] = accu
    else:
        tensors['output_fb'] = output_fb
        tensors['prediction'] = decode_tags

    return tensors
