#-*-encoding:utf-8-*-
import argparse
import sys
import re
import codecs
from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'

class FormatError(Exception):
	pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):
	def __init__(self):
		self.correct = 0			# 'O'를 포함하여 세는 것. accuracy를 위해.
		self.correct_tags = 0		# 'O'를 제외하고 세는 것.
		self.found_correct = 0		#
		self.found_guessed = 0		# number of identified arguments
		self.num_words = 0		# token counter (ignores sentence breaks)

		# counts by type
		self.t_correct_tags = defaultdict(int)
		self.t_found_correct = defaultdict(int)
		self.t_found_guessed = defaultdict(int)

def evaluate_from_list(prediction, gold):
	counts = EvalCounts()
	for p,g in zip(prediction, gold):
		if p == g:
			counts.correct += 1
		if p == g and g not in ['-', 'O']:
			counts.t_correct_tags[g] += 1
			counts.correct_tags += 1

		if g not in ['-', 'O']:
			counts.found_correct += 1
			counts.t_found_correct[g] += 1
		if p not in ['-', 'O']:
			counts.found_guessed += 1
			counts.t_found_guessed[p] += 1
		counts.num_words += 1
	overall, by_type = metrics(counts)
	if counts.num_words > 0:
		return overall.fscore
	else:
		raise ValueError

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
	tp, fp, fn = correct, guessed-correct, total-correct
	p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
	r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
	f = 0 if p + r == 0 else 2 * p * r / (p + r)
	return Metrics(tp, fp, fn, p, r, f)

def metrics(counts):
	c = counts
	overall = calculate_metrics(c.correct_tags, c.found_guessed, c.found_correct)
	by_type = {}
	for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
		by_type[t] = calculate_metrics(c.t_correct_tags[t], c.t_found_guessed[t], c.t_found_correct[t])
	return overall, by_type

def read_prediction(prediction_file):
	'''
	prediction file은 매 행에 문장 단위 argument label이 작성된 상태입니다.
	예) "['ARG0', 'ARG1', '-', ..., '-']\n['ARG0', '-', 'ARG3', ..., '-']
	이 함수는 평가 코퍼스 내 모든 결과를 모아 리턴합니다.
	'''

	predictions = []
	with open(prediction_file, encoding='utf-8') as fp:
		for sentence in fp.read().strip().split('\n'):
			# sentence = "['ARG0', 'ARG1', ..., '-']"
			predictions.extend(eval(sentence))
	# predictions = ['ARG0', 'ARG1', 'ARG2', '-', 'ARG0', ..., ]
	return predictions

def read_ground_truth(ground_truth_file):
	with open(ground_truth_file, encoding='utf-8') as fp:
		ground_truths = [arg.strip() for arg in fp.read().strip().split('\n') if arg.strip()]
	return ground_truths

def evaluation_metrics(prediction_file: str, ground_truth_file: str):
	''' read prediction and ground truth from file '''
	prediction = read_prediction(prediction_file)
	ground_truth = read_ground_truth(ground_truth_file)
	return evaluate_from_list(prediction, ground_truth)

if __name__ == '__main__':
	args = argparse.ArgumentParser() # --prediction 은 inference 결과가 담긴 file 명으로 셋팅합니다. (nsml 내부적으로 셋팅함)
	args.add_argument('--prediction', type=str, default='pred.txt')
	config = args.parse_args()
	# dataset push 를 할때, leaderboard옵션으로 푸시하였으면, 자동으로 test/test_label 의 위치에 test_label 가 존재하므로, 해당위치로 설정합니다.
	test_label_path = '/data/SRL/test/test_label'
	# print the evaluation result
	# evaluation 은 int 또는 float 값으로만 출력합니다.
	try:
		eval_result = evaluation_metrics(config.prediction, test_label_path)
		print (eval_result)
	except:
		# 에러로 인한 0점 처리
	 	print("0")
'''
if __name__ == '__main__':
	print (evaluation_metrics('pred.txt', 'gold.txt'))
'''
