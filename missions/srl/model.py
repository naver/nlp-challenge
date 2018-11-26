#-*-encoding:utf-8-*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import custom_rnncell as rnn
from model_utils import result_to_json
import sys

class Model(object):
	def __init__(self, config):

		self.config = config
		self.lr = config["lr"]
		self.char_dim = config["char_dim"]
		self.char_lstm_dim = config["char_lstm_dim"]
		self.word_lstm_dim = config["word_lstm_dim"]

		self.num_tags = config["num_tags"]
		self.num_chars = config["num_chars"]

		self.global_step = tf.Variable(0, trainable=False)
		self.best_dev_f1 = tf.Variable(0.0, trainable=False)
		self.best_test_f1 = tf.Variable(0.0, trainable=False)
		self.initializer = initializers.xavier_initializer()

		# add placeholders for the model

		self.char_inputs = tf.placeholder(dtype=tf.int32,
											# [batch, word_in_sen, char_in_word]
										  shape=[None, None, config["max_char_length"]],
										  name="ChatInputs")
		self.targets = tf.placeholder(dtype=tf.int32,
									  shape=[None, None],
									  name="Targets")
		# dropout keep prob
		self.dropout = tf.placeholder(dtype=tf.float32,
									  name="Dropout")

		used = tf.sign(tf.abs(self.char_inputs)) #존재하는 곳에 1인 mask
		char_length = tf.reduce_sum(used, reduction_indices=2)
		word_length = tf.reduce_sum(tf.sign(char_length), reduction_indices=1)
		self.word_lengths = tf.cast(word_length, tf.int32)
		self.batch_size = tf.shape(self.char_inputs)[0]
		self.word_num_steps = tf.shape(self.char_inputs)[-2]

		# embeddings for chinese character and segmentation representation
		embedding = self.embedding_layer(self.char_inputs)

		# apply dropout before feed to lstm layer
		embedding = tf.nn.dropout(embedding, self.dropout)

		# position-encoding and bi-directional lstm layer
		word_encoded = self.get_word_representation(embedding)
		lstm_outputs = self.biLSTM_layer(word_encoded, self.word_lstm_dim, self.word_lengths)

		# logits for tags
		self.logits = self.project_layer(lstm_outputs)

		# loss of the model
		self.loss = self.loss_layer(self.logits, self.word_lengths)

		with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
			optimizer = self.config["optimizer"]
			if optimizer == "sgd":
				self.opt = tf.train.GradientDescentOptimizer(self.lr)
			elif optimizer == "adam":
				self.opt = tf.train.AdamOptimizer(self.lr)
			elif optimizer == "adgrad":
				self.opt = tf.train.AdagradOptimizer(self.lr)
			else:
				raise KeyError

			# apply grad clip to avoid gradient explosion
			grads_vars = self.opt.compute_gradients(self.loss)
			capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
								 for g, v in grads_vars]
			self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

		# saver of the model
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

	def get_word_representation(self, embedding):
		position_encoding_mat = self._position_encoding(self.config["max_char_length"], self.char_dim)
		position_encoded = tf.reduce_sum(embedding * position_encoding_mat, 2)
		return position_encoded
	
	def _position_encoding(self, sentence_size, embedding_size):
		encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
		ls = sentence_size+1
		le = embedding_size+1
		for i in range(1, le):
			for j in range(1, ls):
				encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
		encoding = 1 + 4 * encoding / embedding_size / sentence_size
		return np.transpose(encoding)

	def embedding_layer(self, char_inputs, name=None):
		"""
		:param char_inputs: one-hot encoding of sentence
		:return: [1, num_steps, embedding size], 
		"""

		with tf.variable_scope("char_embedding" if not name else name, reuse=tf.AUTO_REUSE), tf.device('/cpu:0'):
			self.char_lookup = tf.get_variable(
					name="char_embedding",
					shape=[self.num_chars, self.char_dim],
					initializer=self.initializer)
			embed = tf.nn.embedding_lookup(self.char_lookup, char_inputs)
		return embed

	def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
		"""
		:param lstm_inputs: [batch_size, num_steps, emb_size] 
		:return: [batch_size, num_steps, 2*lstm_dim] 
		"""
		with tf.variable_scope("char_BiLSTM" if not name else name, reuse=tf.AUTO_REUSE):
			lstm_cell = {}
			for direction in ["forward", "backward"]:
				with tf.variable_scope(direction, reuse=tf.AUTO_REUSE):
					lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
						lstm_dim,
						use_peepholes=True,
						initializer=self.initializer,
						state_is_tuple=True)
			outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
				lstm_cell["forward"],
				lstm_cell["backward"],
				lstm_inputs,
				dtype=tf.float32,
				sequence_length=lengths)
		return tf.concat(outputs, axis=2)

	def project_layer(self, lstm_outputs, name=None):
		"""
		hidden layer between lstm layer and logits
		:param lstm_outputs: [batch_size, num_steps, emb_size] 
		:return: [batch_size, num_steps, num_tags]
		"""
		with tf.variable_scope("project"  if not name else name, reuse=tf.AUTO_REUSE):
			with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
				W = tf.get_variable("W", shape=[self.word_lstm_dim*2, self.word_lstm_dim],
									dtype=tf.float32, initializer=self.initializer)

				b = tf.get_variable("b", shape=[self.word_lstm_dim], dtype=tf.float32,
									initializer=tf.zeros_initializer())
				output = tf.reshape(lstm_outputs, shape=[-1, self.word_lstm_dim*2])
				hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

			# project to score of tags
			with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
				W = tf.get_variable("W", shape=[self.word_lstm_dim, self.num_tags],
									dtype=tf.float32, initializer=self.initializer)

				b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
									initializer=tf.zeros_initializer())

				pred = tf.nn.xw_plus_b(hidden, W, b)

			return tf.reshape(pred, [-1, self.word_num_steps, self.num_tags])

	def loss_layer(self, project_logits, lengths, name=None):
		"""
		calculate crf loss
		:param project_logits: [1, num_steps, num_tags]
		:return: scalar loss
		"""
		with tf.variable_scope("crf_loss"  if not name else name, reuse=tf.AUTO_REUSE):
			small = -1000.0
			# pad logits for crf loss
			start_logits = tf.concat(
				[small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
			pad_logits = tf.cast(small * tf.ones([self.batch_size, self.word_num_steps, 1]), tf.float32)
			logits = tf.concat([project_logits, pad_logits], axis=-1)
			logits = tf.concat([start_logits, logits], axis=1)
			targets = tf.concat(
				[tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

			self.trans = tf.get_variable(
				"transitions",
				shape=[self.num_tags + 1, self.num_tags + 1],
				initializer=self.initializer)
			log_likelihood, self.trans = crf_log_likelihood(
				inputs=logits,
				tag_indices=targets,
				transition_params=self.trans,
				sequence_lengths=lengths+1)
			return tf.reduce_mean(-log_likelihood)

	def create_feed_dict(self, is_train, batch):
		"""
		:param is_train: Flag, True for train batch
		:param batch: list train/evaluate data 
		:return: structured data to feed
		"""
		_, chars, tags = batch
		feed_dict = {
			self.char_inputs: np.array(chars),
			self.dropout: 1.0,
		}
		'''
		print ('chars')
		print (chars)
		print ('after chars')
		print (feed_dict[self.char_inputs])
		print 
		'''
		if is_train:
			feed_dict[self.targets] = np.asarray(tags)
			feed_dict[self.dropout] = self.config["dropout_keep"]
			'''
			print ('tags')
			print (tags)
			print ('after tags')
			print (feed_dict[self.targets])
			print
			'''
		return feed_dict

	def run_step(self, sess, is_train, batch):
		"""
		:param sess: session to run the batch
		:param is_train: a flag indicate if it is a train batch
		:param batch: a dict containing batch data
		:return: batch result, loss of the batch or logits
		"""
		feed_dict = self.create_feed_dict(is_train, batch)
		
		if is_train:
			global_step, loss, _ = sess.run(
				[self.global_step, self.loss, self.train_op],
				feed_dict)
			return global_step, loss
		else:
			lengths, logits = sess.run([self.word_lengths, self.logits], feed_dict)
			return lengths, logits

	def decode(self, logits, lengths, matrix):
		"""
		:param logits: [batch_size, num_steps, num_tags]float32, logits
		:param lengths: [batch_size]int32, real length of each sequence
		:param matrix: transaction matrix for inference
		:return:
		"""
		# inference final labels usa viterbi Algorithm
		paths = []
		small = -1000.0
		start = np.asarray([[small]*self.num_tags +[0]])
		for score, length in zip(logits, lengths):
			score = score[:length]
			pad = small * np.ones([length, 1])
			logits = np.concatenate([score, pad], axis=1)
			logits = np.concatenate([start, logits], axis=0)
			path, _ = viterbi_decode(logits, matrix)

			paths.append(path[1:])
		return paths

	def evaluate_model(self, sess, data_manager, id_to_tag):
		"""
		:param sess: session  to run the model 
		:param data: list of data
		:param id_to_tag: index to tag name
		:return: evaluate result
		"""
		results = []
		trans = self.trans.eval(session=sess)
		for batch in data_manager.iter_batch():
			strings = batch[0]
			tags = batch[-1]
			lengths, scores = self.run_step(sess, False, batch)
			batch_paths = self.decode(scores, lengths, trans)
			for i in range(len(strings)):
				result = []
				string = strings[i][:lengths[i]]
				gold = [id_to_tag[int(x)] for x in tags[i][:lengths[i]]]
				pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]
				for char, gold, pred in zip(string, gold, pred):
					result.append(" ".join([''.join(char), gold, pred]))
				results.append(result)
		return results

	def evaluate_lines(self, sess, inputs, id_to_tag):
		trans = self.trans.eval(session=sess)
		lengths, scores = self.run_step(sess, False, inputs)
		batch_paths = self.decode(scores, lengths, trans)
		total_tags = [[id_to_tag[idx] for idx in path] for path in batch_paths]
		return [(0.0,tag) for tag in total_tags]
