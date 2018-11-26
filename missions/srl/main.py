# encoding=utf8
import os
import pickle
import sys
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import char_mapping, tag_mapping, prepare_dataset
from model_utils import get_logger, make_path, create_model, save_model, print_config, save_config, load_config, test_srl
from data_utils import inputs_from_sentences, BatchManager
from data_loader import data_loader

try:
	import nsml
	from nsml import DATASET_PATH, IS_ON_NSML
except ImportError as e:
	IS_ON_NSML = False

def parse_args():
	flags = tf.app.flags
	# configurations for the model
	flags.DEFINE_integer("seg_dim",			0,		"Embedding size for segmentation, 0 if not used")
	flags.DEFINE_integer("char_dim",		100,	"Embedding size for characters")
	flags.DEFINE_integer("char_lstm_dim",	100,	"Num of hidden units in char LSTM")
	flags.DEFINE_integer("word_lstm_dim",	100,	"Num of hidden units in word LSTM")
	flags.DEFINE_integer("max_char_length",	8,		"max number of character in word")
	flags.DEFINE_integer("max_word_length",	95,		"number of word")
	flags.DEFINE_integer("num_tags",		29,		"number of tags")
	flags.DEFINE_integer("num_chars",		8000,	"number of chars")

	# configurations for training
	flags.DEFINE_float("clip",			5,			"Gradient clip")
	flags.DEFINE_float("dropout",		0.5,		"Dropout rate")
	flags.DEFINE_float("lr",			0.001,		"Initial learning rate")
	flags.DEFINE_string("optimizer",	"adam",		"Optimizer for training")
	flags.DEFINE_boolean("lower",		True,		"Wither lower case")
	flags.DEFINE_integer("batch_size",	20,			"batch size")
	flags.DEFINE_integer("patience",	5,			"Patience for the validation-based early stopping")

	flags.DEFINE_integer("max_epoch",	5,		"maximum training epochs")
	flags.DEFINE_integer("steps_check", 100,		"steps per checkpoint")
	flags.DEFINE_string("ckpt_path",	"ckpt",		 "Path to save model")
	flags.DEFINE_string("summary_path", "summary",		"Path to store summaries")
	flags.DEFINE_string("log_file",		"train.log",	"File for log")
	flags.DEFINE_string("map_file",		"maps.pkl",		"file for maps")
	flags.DEFINE_string("vocab_file",	"vocab.json",	"File for vocab")
	flags.DEFINE_string("config_file",	"config_file",	"File for config")
	flags.DEFINE_string("script",		"conlleval",	"evaluation script")
	flags.DEFINE_string("result_path",	"result",		"Path for results")

	# reserved for NSML
	flags.DEFINE_integer("pause",		0,			"Pause")
	flags.DEFINE_string("mode",			"train",	"Train/Test mode")

	# dataset
	if IS_ON_NSML:
		flags.DEFINE_string("DATASET_PATH",	nsml.DATASET_PATH, "path for dataset")
	else:
		flags.DEFINE_string("DATASET_PATH",	'./data/', "path for dataset")

	FLAGS = tf.app.flags.FLAGS
	assert FLAGS.clip < 5.1, "gradient clip should't be too much"
	assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
	assert FLAGS.lr > 0, "learning rate must larger than zero"
	assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]

	return FLAGS


# config for the model
def config_model():
	config = OrderedDict()
	config["num_chars"] = FLAGS.num_chars
	config["char_dim"] = FLAGS.char_dim
	config["num_tags"] = FLAGS.num_tags
	config["seg_dim"] = FLAGS.seg_dim
	config["char_lstm_dim"] = FLAGS.char_lstm_dim
	config["word_lstm_dim"] = FLAGS.word_lstm_dim
	config["batch_size"] = FLAGS.batch_size

	config["clip"] = FLAGS.clip
	config["dropout_keep"] = 1.0 - FLAGS.dropout
	config["optimizer"] = FLAGS.optimizer
	config["lr"] = FLAGS.lr
	config["lower"] = FLAGS.lower
	config["max_char_length"] = FLAGS.max_char_length
	config["max_word_length"] = FLAGS.max_word_length

	return config

"""
매 epoch에 validation/test set의 평가를 수행하고 그 결과를 출력한다.
"""
def evaluate(sess, model, name, data, id_to_tag, logger):
	logger.info("evaluate:{}".format(name))
	
	srl_results = model.evaluate_model(sess, data, id_to_tag)
	eval_lines = test_srl(srl_results, FLAGS.result_path)
	for line in eval_lines:
		logger.info(line.strip())
	f1 = float(eval_lines[1].strip().split()[-1])

	if name == "dev":
		best_test_f1 = model.best_dev_f1.eval(session=sess)
		if f1 > best_test_f1:
			tf.assign(model.best_dev_f1, f1).eval(session=sess)
			logger.info("new best dev f1 score:{:>.3f}".format(f1))
		return f1 > best_test_f1, f1
	elif name == "test":
		best_test_f1 = model.best_test_f1.eval(session=sess)
		if f1 > best_test_f1:
			tf.assign(model.best_test_f1, f1).eval(session=sess)
			logger.info("new best test f1 score:{:>.3f}".format(f1))
		return f1 > best_test_f1, f1


def bind_model(sess, FLAGS):

	def save(dir_path, *args):

		os.makedirs(dir_path, exist_ok=True)
		saver = tf.train.Saver()
		saver.save(sess, os.path.join(dir_path, 'model'))

		with open(os.path.join(dir_path,FLAGS.map_file), "wb") as f:
			pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)

	def load(dir_path, *args):
		global char_to_id
		global id_to_tag

		config = load_config(FLAGS.config_file)
		logger = get_logger(FLAGS.log_file)
		tf.get_variable_scope().reuse_variables()

		with open(os.path.join(dir_path,FLAGS.map_file), "rb") as f:
			char_to_id, _, __, id_to_tag = pickle.load(f)

		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(dir_path)
		if ckpt and ckpt.model_checkpoint_path:
			checkpoint = os.path.basename(ckpt.model_checkpoint_path)
			saver.restore(sess, os.path.join(dir_path, checkpoint))
		else:
			raise NotImplemented('No checkpoint found!')
		print ('model loaded!')

	def infer(sentences, **kwargs):
		config = load_config(FLAGS.config_file)
		logger = get_logger(FLAGS.log_file)
		
		reformed_sentences = [' '.join(sen[1]) for sen in sentences]
		result = model.evaluate_lines(sess, inputs_from_sentences(reformed_sentences, char_to_id, FLAGS.max_char_length), id_to_tag)
		'''
		result = [
			       (0.0, ['ARG0', 'ARG3', '-']),
				   (0.0, ['ARG0', 'ARG1', '-'])
				 ]
		# evaluate_lines 함수는 문장 단위 분석 결과를 내어줍니다.
		# len(result) : 문장의 갯수, 따라서 위 예제는 두 문장의 결과입니다.
		# result[0] : 첫번째 문장의 분석 결과, result[1] : 두번째 문장의 분석 결과.
		
		# 각 문장의 분석 결과는 다시 (prob, [labels])로 구성됩니다.
		# prob에 해당하는 자료는 이번 task에서 사용하지 않습니다. 따라서 그 값이 결과에 영향을 미치지 않습니다.
		# [labels]는 각 어절의 분석 결과를 담고 있습니다. 따라서 다음과 같이 구성됩니다.
		## ['첫번째 어절의 분석 결과', '두번째 어절의 분석 결과', ...]
		# 예를 들면 위 주어진 예제에서 첫번째 문장의 첫번째 어절은 'ARG0'을, 첫번째 문장의 두번째 어절은 'ARG3'을 argument label로 가집니다.

		### 주의사항 ###
		# 모든 어절의 결과를 제출하여야 합니다.
		# 어절의 순서가 지켜져야 합니다. (첫번째 어절부터 순서대로 list 내에 위치하여야 합니다.)
		'''
		return result

	if IS_ON_NSML:
		nsml.bind(save=save, load=load, infer=infer)

def train(sess):
	if not IS_ON_NSML:
		with open(FLAGS.map_file, "wb") as f:
			pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)

	steps_per_epoch = train_manager.len_data
	early_stop = 0

	logger.info("start training")
	loss = []
	for epoch in range(FLAGS.max_epoch):
		for batch in train_manager.iter_batch(shuffle=True):
			step, batch_loss = model.run_step(sess, True, batch)
			loss.append(batch_loss)
			if step % FLAGS.steps_check == 0:
				iteration = step // steps_per_epoch + 1
				logger.info("iteration:{} step:{}/{}, "
							"loss:{:>9.6f}".format(iteration, step%steps_per_epoch, \
									steps_per_epoch, np.mean(loss)))

		best, dev_f1 = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)

		# early stopping
		if best: early_stop = 0
		else:
			early_stop += 1
			if early_stop > FLAGS.patience: break

		# save model
		if best:
			save_model(sess, model, FLAGS.ckpt_path, logger)
		if IS_ON_NSML:
			nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=FLAGS.max_epoch, train__loss=float(np.mean(loss)), valid__f1score=dev_f1)
			nsml.save(epoch)
		loss = []
	

def evaluate_cli(model):
	config = load_config(FLAGS.config_file)
	logger = get_logger(FLAGS.log_file)
	with open(FLAGS.map_file, "rb") as f:
		char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
	while True:
		line = input("문장을 입력하세요.:")
		result = model.evaluate_lines(sess, inputs_from_sentences([line], char_to_id, FLAGS.max_char_length), id_to_tag)
		print(result)


if __name__ == '__main__':
	FLAGS = parse_args()

	# tensorflow config
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	sess = tf.Session(config=tf_config)

	# model config
	make_path(FLAGS)
	config = config_model()
	save_config(config, FLAGS.config_file)
	log_path = os.path.join("log", FLAGS.log_file)
	logger = get_logger(log_path)
	print_config(config, logger)

	# create model
	model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)

	bind_model(sess, FLAGS)
	if FLAGS.pause and IS_ON_NSML:
		nsml.paused(scope=locals())

	if FLAGS.mode == 'train':
		dataset = data_loader(FLAGS.DATASET_PATH)
		train_dataset, dev_dataset = dataset[:-3000], dataset[-3000:]

		# create dictionary for word
		_c, char_to_id, id_to_char = char_mapping(train_dataset, FLAGS.lower)

		# create a dictionary and a mapping for tags
		# 태그 빈도, tag to id dictionary, id to tag dictionary
		_, tag_to_id, id_to_tag = tag_mapping(train_dataset)

		# prepare data, get a collection of list containing index
		train_data = prepare_dataset(train_dataset, char_to_id, tag_to_id)
		dev_data = prepare_dataset(dev_dataset, char_to_id, tag_to_id)
		print("%i / %i  sentences in train / dev " % (
			len(train_data), len(dev_data)))

		train_manager = BatchManager(train_data, FLAGS.batch_size, FLAGS.max_char_length)
		dev_manager = BatchManager(dev_data, 100, FLAGS.max_char_length)
		train(sess)
	elif FLAGS.mode == 'local_test_cli':
		evaluate_cli(model)
