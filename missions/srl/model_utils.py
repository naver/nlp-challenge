#-*-encoding:utf-8-*-
import os
import json
import shutil
import logging

import tensorflow as tf
from conlleval import return_report


def get_logger(log_file):
	logger = logging.getLogger(log_file)
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler(log_file)
	fh.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	ch.setFormatter(formatter)
	fh.setFormatter(formatter)
	logger.addHandler(ch)
	logger.addHandler(fh)
	return logger


def test_srl(results, path):
	"""
	perl script를 이용해 평가
	"""
	output_file = os.path.join(path, "srl_predict.utf8")
	with open(output_file, "w", encoding='utf-8') as f:
		to_write = []
		for block in results:
			for line in block:
				to_write.append(line + "\n")
			to_write.append("\n")

		f.writelines(to_write)
	eval_lines = return_report(output_file)
	return eval_lines


def print_config(config, logger):
	"""
	Print configuration of the model
	"""
	for k, v in config.items():
		logger.info("{}:\t{}".format(k.ljust(15), v))


def make_path(params):
	"""
	Make folders for training and evaluation
	"""
	if not os.path.isdir(params.result_path):
		os.makedirs(params.result_path)
	if not os.path.isdir(params.ckpt_path):
		os.makedirs(params.ckpt_path)
	if not os.path.isdir("log"):
		os.makedirs("log")


def save_config(config, config_file):
	"""
	Save configuration of the model
	parameters are stored in json format
	"""
	with open(config_file, "w", encoding="utf8") as f:
		json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
	"""
	Load configuration of the model
	parameters are stored in json format
	"""
	with open(config_file, encoding="utf8") as f:
		return json.load(f)


def save_model(sess, model, path, logger):
	checkpoint_path = os.path.join(path, "srl.ckpt")
	model.saver.save(sess, checkpoint_path)
	logger.info("model saved")


def create_model(session, Model_class, path, config, logger):
	# create model, reuse parameters if exists
	model = Model_class(config)

	ckpt = tf.train.get_checkpoint_state(path)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		logger.info("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model


def result_to_json(string, tags):
	item = {"string": string, "entities": []}
	idx = 0
	for char, tag in zip(string.split(), tags):
		if tag not in ['-', 'O']:
			item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag})
		idx += 1
	return item


