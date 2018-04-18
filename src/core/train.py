# -*- coding:utf-8 -*-
"""
@File      : train.py
@Software  : OCR
@Time      : 2018/3/30 17:22
@Author    : yubb
"""
import os
import numpy as np
import tensorflow as tf
import ops.lenet5 as network
from data.dataset import TextDataSet, load_data


tf.app.flags.DEFINE_float('margin', 0.3, "*margin.")
# 务必为3的倍数: 每次抽取30个字进行
tf.app.flags.DEFINE_integer('batch_size', 3, "*batch size.")
tf.app.flags.DEFINE_integer('examples_num', 3, "*examples num.")
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', "*Name of model.")
tf.app.flags.DEFINE_string('dest', './data', "*Location of dataset.")
tf.app.flags.DEFINE_string('model_save_path', './data', "*model save path.")
tf.app.flags.DEFINE_float('regularization_rate', 0.0001, "*regularization rate.")
tf.app.flags.DEFINE_float('moving_average_decay', 0.99, "*moving average decay.")
tf.app.flags.DEFINE_float('learning_rate_base', 0.8, "*learning rate base.")
tf.app.flags.DEFINE_float('learning_rate_decay', 0.99, "*learning rate decay.")
tf.app.flags.DEFINE_integer('embeddings_size', 128, "*examples num.")
FLAGS = tf.app.flags.FLAGS


def train():
    dataset = TextDataSet(FLAGS.dest)
    episodes = dataset.dataset_size // FLAGS.batch_size
    inputs = tf.placeholder(tf.float32, [FLAGS.examples_num,
                                         network.IMAGE_SIZE,
                                         network.IMAGE_SIZE,
                                         network.NUM_CHANNELS], name='inputs')
    regularizer = tf.contrib.layers.l2_regularizer(FLAGS.regularization_rate)
    prelogits = network.inference(inputs, is_training=True, regularizer=regularizer)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, FLAGS.embeddings_size]), 3, 1)

    # 滑动平均
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base, global_step, episodes,
                                               FLAGS.learning_rate_decay)
    tf.summary.scalar('learning_rate', learning_rate)

    # 计算loss
    triplet_loss = network.triplet_loss(anchor, positive, negative, FLAGS.margin)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

    # 计算梯度
    opt = tf.train.AdagradOptimizer(learning_rate)
    grads = opt.compute_gradients(total_loss, tf.global_variables())
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    emb_array = np.zeros((dataset.dataset_size, FLAGS.examples_num))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        model_path = os.path.join(FLAGS.model_save_path, 'checkpoint')
        if os.path.exists(model_path):
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, os.path.expanduser(ckpt.model_checkpoint_path))
                print("Checkpoint File Exists.")
                quit()
        for step in range(episodes):
            ds, dirs, labels = dataset.next_batch(FLAGS.batch_size)
            skip = len(dirs) // len(ds)
            cur = 0
            for dt in dirs:
                label = labels[cur // skip]
                batch = load_data(dt)
                data = np.reshape(batch, (FLAGS.examples_num, network.IMAGE_SIZE, network.IMAGE_SIZE, network.NUM_CHANNELS))
                embs, _, loss_val, step = sess.run([embeddings, train_op, total_loss, global_step],
                                                   feed_dict={inputs: data})
                print("label =", label, "emb = ", embs, "step = ", step, "loss = ", loss_val)
                cur += 1
            saver.save(sess, model_path, global_step=global_step)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()

