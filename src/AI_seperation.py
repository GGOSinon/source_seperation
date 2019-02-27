import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.slim as slim

BATCH_SIZE = 32
BETA = 0.01

def dense(x, W, b, activation = 'relu', use_bn = True, keep_prob = 1.0):
	x = tf.add(tf.matmul(x, W), b)
	if use_bn: x = tf.layers.batch_normalization(x)
	if activation == 'x': res = x
	if activation == 'sigmoid': res = tf.nn.sigmoid(x)
	if activation == 'relu': res = tf.nn.relu(x)
	res = tf.nn.dropout(res, keep_prob = keep_prob)
	return res

def conv2D(x, W, b, strides = 1, activation = 'relu', use_bn = True):
	x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	if use_bn: x = tf.layers.batch_normalization(x)
	if activation == 'relu': return tf.nn.relu(x)

class Model:
	
	def __init__(self, trainData):
		self.trainData = trainData
		# Define Q networks
		self.weights = {
			'wd1': tf.get_variable('wd1', [2049 * 10, 500], initializer = tf.contrib.layers.xavier_initializer()),
			'wd2': tf.get_variable('wd2', [500, 500], initializer = tf.contrib.layers.xavier_initializer()),
			'wd3': tf.get_variable('wd3', [500, 2049 * 10], initializer = tf.contrib.layers.xavier_initializer()),
		}
		self.biases = {
			'bd1': tf.get_variable('bd1', [500], initializer = tf.contrib.layers.xavier_initializer()),
			'bd2': tf.get_variable('bd2', [500], initializer = tf.contrib.layers.xavier_initializer()),
			'bd3': tf.get_variable('bd3', [2049 * 10], initializer = tf.contrib.layers.xavier_initializer()),
		}
	
		self.x = tf.placeholder(tf.float32, [None, 2049, 10])
		self.y_hat = tf.placeholder(tf.float32, [None, 2049, 10])
		self.keep_prob = tf.placeholder(tf.float32)

		self.y = self.forward(self.x, self.weights, self.biases)
		self.loss = tf.reduce_mean(tf.square(self.y - self.y_hat))
		self.regularizer = tf.Variable(0.0, trainable = False)
		for keys in self.weights:
			if keys == 'wd3': continue
			self.regularizer += tf.nn.l2_loss(self.weights[keys])
		
		self.regloss = tf.reduce_mean(self.loss + BETA * self.regularizer)
		
		self.global_step = tf.Variable(0, trainable = False)
		#self.learning_rate = tf.train.exponential_decay(0.01, self.global_step, 1000, 0.99, staircase = True)
		self.learning_rate = tf.Variable(0.001)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.regloss, global_step = self.global_step)
		
		config = tf.ConfigProto()
		#config.intra_op_parallelism_threads = 8
		#config.inter_op_parallelism_threads = 8
		self.sess = tf.InteractiveSession(config = config)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(tf.global_variables())
				
	def forward(self, x, weights, biases):
		x = tf.reshape(x, [-1, 2049 * 10])
		x = dense(x, weights['wd1'], biases['bd1'], activation = 'relu', use_bn = True, keep_prob = self.keep_prob)
		x = dense(x, weights['wd2'], biases['bd2'], activation = 'relu', use_bn = True, keep_prob = self.keep_prob)
		x = dense(x, weights['wd3'], biases['bd3'], activation = 'relu', use_bn = False, keep_prob = self.keep_prob)
		x = tf.reshape(x, [-1, 2049, 10])
		return x

	def train(self):
		x, y_hat = [], []
		for _ in range(BATCH_SIZE):
			pos = random.randrange(0, len(self.trainData))
			x.append(self.trainData[0][pos][0])
			y_hat.append(self.trainData[0][pos][0])

		self.sess.run(self.opt, feed_dict = {self.x: x, self.y_hat: y_hat, self.keep_prob: 0.5})
		loss, regloss, learning_rate = self.sess.run([self.loss, self.regloss, self.learning_rate], feed_dict = {self.x: x, self.y_hat: y_hat, self.keep_prob: 1.0})
		return loss, regloss, learning_rate

	def save(self, name):
		self.saver.save(self.sess, name)

	def load(self, name):
		self.saver.restore(self.sess, name)
	
	
