#-*-encoding:utf-8-*-
import os, sys, re, codecs, json

from data_utils import create_dico, create_mapping, zero_digits

def _load_conll_file(path):
	"""
	파일을 읽어 문장 단위 list로 리턴한다.
	"""
	sentences = []
	sentence = [[], [], []]

	with codecs.open(path, 'r', 'utf-8') as fp:
		for line in fp: #라인 단위로
			line = line.rstrip()

			if not line: #문장이 끝나면
				if len(sentence) > 0:
					sentences.append(sentence) #보관한다.
					sentence = []
			else: #문장이 진행 중이면
				word = line.split()
				sentence.append(word)
		if len(sentence) > 0:
			sentences.append(sentence)
	return sentences

def char_mapping(sentences, lower):
	"""
	음절 사전을 구축한다.
	"""
	if lower:
		chars = [[[char for char in word.lower()] for word in s[1]] for s in sentences]
	else:
		chars = [[[char for char in word] for word in s[1]] for s in sentences]
	dico = create_dico(chars)
	dico["<PAD>"] = 10000001
	dico['<UNK>'] = 10000000
	char_to_id, id_to_char = create_mapping(dico)
	print("Found %i unique chars" % (len(dico)))
	return dico, char_to_id, id_to_char


def tag_mapping(sentences):
	"""
	Create a dictionary and a mapping of tags, sorted by frequency.
	"""
	tags = [[tag for tag in s[2]] for s in sentences]
	dico = create_dico(tags)
	tag_to_id, id_to_tag = create_mapping(dico)
	print("Found %i unique tags" % len(dico))
	return dico, tag_to_id, id_to_tag


def prepare_dataset(dataset, char_to_id, tag_to_id, train=True):
	"""
	데이터셋 전처리를 수행한다.
	return : list of list of dictionary
	dictionry
		- word indices
		- word char indices
		- tag indices
	"""
	none_index = tag_to_id["O"] if "O" in tag_to_id else tag_to_id["-"]

	data = []
	for idx, sen in enumerate(dataset):
		#sen : [['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], ['나는', '다른', '방배들과', '한', '다발씩', '묶여', '컴컴한', '금융기관', '속으로', '옮겨졌다.'], ['ARG0', '-', '-', '-', '-', '-', '-', '-', 'ARG3', '-']]
		words = sen[1]
		chars = [[char_to_id[c if c in char_to_id else '<UNK>']
				 for c in word] for word in sen[1]]
		if train:
			tag_ids = [tag_to_id[l] for l in sen[2]]
		else:
			tag_ids = [none_index for _ in chars]
		data.append([words, chars, tag_ids])

	return data


