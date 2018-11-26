#-*-encoding:utf-8-*-
import os, logging

def test_data_loader(root_path):
	"""
	문장을 리턴한다.
	return: list of sentence
	"""
	data_path = os.path.join(root_path, 'test', 'test_data')
	return _load_data(data_path, is_train=False)

def local_test_data_loader(root_path):
	"""
	문장을 리턴한다.
	return: list of sentence
	"""
	data_path = os.path.join(root_path, 'test', 'test_data')
	return _load_data(data_path, is_train=True)

def _load_data(data_path, is_train=False):
	"""
	파일을 읽어 문장 단위 list로 리턴한다.
	"""
	sentences = []
	sentence = [[], [], []]

	with open(data_path, encoding='utf-8') as fp:
		contents = fp.read().strip()
		for line in contents.split('\n'):
			if line == '':
				sentences.append(sentence)
				sentence = [[], [], []]
			else:
				idx, eojeol, label = line.split('\t')
				sentence[0].append(idx)
				sentence[1].append(eojeol)
				sentence[2].append(label)
	return sentences

	'''
	# dataset
	with open(data_path, encoding='utf-8') as fp:
		contents = fp.read().strip()
		for line in contents.split('\n'):
			if line == '':
				sentences.append(sentence)
				sentence = [[], [], []]
			else:
				splitted = line.split('\t')
				idx = splitted[0]
				eojeol = splitted[1]
				sentence[0].append(idx)
				sentence[1].append(eojeol)

	# label
	if is_train:
		with open(label_path, encoding='utf-8') as fp:
			labels = fp.read().strip().split('\n\n')
			for idx in range(len(sentences)):
				sentences[idx][2] = labels[idx].split('\n')
	else:
		# test dataset은 label이 존재하지 않는다.
		for idx in range(len(sentences)):
			sentences[idx][2] = ['-']*len(sentences[idx][1])

	return sentences
	'''

def data_loader(root_path):
	data_path = os.path.join(root_path, 'train', 'train_data')
	return _load_data(data_path, is_train=True)

