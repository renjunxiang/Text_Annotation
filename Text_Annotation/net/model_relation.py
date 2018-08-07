import tensorflow as tf

input_embedding = tf.placeholder(shape=[None, 2, 4], dtype=tf.float32)
input_indexes = tf.placeholder(shape=[None, 2], dtype=tf.float32)
output_targets = tf.placeholder(shape=[None, 5], dtype=tf.float32)


def relation_extraction(input_embedding=None,
                        input_indexes=None,
                        output_targets=None,
                        num_units=128,
                        batchsize=64,
                        num_tags=10):
    '''

    :param input_embedding: 文本embedding后的向量序列
    :param input_indexes: 实体位置索引
    :param output_targets: 关系类别
    :param num_units: embedding维度
    :param batchsize:
    :param num_tags: 关系类别种类
    :return:
    '''
    tensors={}

    with tf.variable_scope('relation'):
        input_indexes = tf.expand_dims(input=input_indexes, dim=2)
        input_indexes = tf.concat([input_indexes] * (num_units * 2), axis=2)
        entity_pair = tf.multiply(input_embedding, input_indexes)
        entity_pair = tf.reduce_mean(entity_pair, axis=1)

    with tf.variable_scope('classify'):
        dense = tf.layers.dense(inputs=entity_pair, units=500, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=num_tags)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=output_targets,
                                                              logits=logits)
        loss = tf.reduce_mean(loss)

    if batchsize > 1:
        train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        tensors['train_op'] = train_op
        tensors['loss'] = loss
        tensors['prediction'] = logits
    else:
        tensors['prediction'] = logits
        tensors['loss'] = loss

    return tensors
