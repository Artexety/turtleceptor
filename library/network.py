import tensorflow as tf

from .config import *


def weight_variable(shape):
	x = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(x, name="weights")


def bias_variable(shape):
	x = tf.constant(0., shape = shape)
	return tf.Variable(x, name="bias")


def conv2d(x, W, stride_h, stride_w):
	return tf.nn.conv2d(x, W, strides = [1, stride_h, stride_w, 1], padding = "SAME")


def variable_summaries(var):
    """ used for tensorBoard visualization """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)

    tf.summary.scalar('mean', mean)

    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


class QNetwork(object):

    def __init__(self, session) -> None:
        with tf.name_scope("conv_1"):
            w_conv1 = weight_variable([10, 14, K_IMAGE_HIST_SIZE, 32])
            variable_summaries(w_conv1)
            b_conv1 = bias_variable([32])

        with tf.name_scope("conv_2"):
            w_conv2 = weight_variable([4, 4, 32, 64])
            variable_summaries(w_conv2)
            b_conv2 = bias_variable([64])

        with tf.name_scope("conv_3"):
            w_conv3 = weight_variable([3, 3, 64, 64])
            variable_summaries(w_conv3)
            b_conv3 = bias_variable([64])

        with tf.name_scope("fc_value"):
            w_value = weight_variable([K_H_SIZE, 512])
            variable_summaries(w_value)
            b_value = bias_variable([512])

        with tf.name_scope("fc_advantage"):
            w_adv = weight_variable([K_H_SIZE, 512])
            variable_summaries(w_adv)
            b_adv = bias_variable([512])

        with tf.name_scope("fc_value_out"):
            w_value_out = weight_variable([512, 1])
            variable_summaries(w_value_out)
            b_value_out = bias_variable([1])

        with tf.name_scope("fc_advantage_out"):
            w_adv_out = weight_variable([512, K_NUM_VALID_ACTIONS])
            variable_summaries(w_adv_out)
            b_adv_out = bias_variable([K_NUM_VALID_ACTIONS])

        self.state = tf.placeholder("float", [None, K_DEPTH_IMAGE_HEIGHT, K_DEPTH_IMAGE_WIDTH, K_IMAGE_HIST_SIZE])

        h_conv1 = tf.nn.relu(conv2d(self.state, w_conv1, 8, 8) + b_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, 2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, 1, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, K_H_SIZE])
        h_fc_value = tf.nn.relu(tf.matmul(h_conv3_flat, w_value) + b_value)
        value = tf.matmul(h_fc_value, w_value_out) + b_value_out
        h_fc_adv = tf.nn.relu(tf.matmul(h_conv3_flat, w_adv) + b_adv)		
        advantage = tf.matmul(h_fc_adv, w_adv_out) + b_adv_out

        adv_average = tf.expand_dims(tf.reduce_mean(advantage, axis = 1), axis = 1)
        adv_identifiable = tf.subtract(advantage, adv_average)
        self.readout = tf.add(value, adv_identifiable)

        self.a = tf.placeholder("float", [None, K_NUM_VALID_ACTIONS])
        self.y = tf.placeholder("float", [None])
        
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), axis = 1)
        self.error = tf.square(self.y - self.readout_action)
        self.cost = tf.reduce_mean(self.error)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)