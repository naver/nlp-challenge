#-*-encoding:utf-8-*-
# Python version of the evaluation script from CoNLL'00-
# Originates from: https://github.com/spyysalo/conlleval.py

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

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

def parse_args(argv):
	import argparse
	parser = argparse.ArgumentParser(
		description='evaluate tagging results using CoNLL criteria',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	arg = parser.add_argument
	arg('-b', '--boundary', metavar='STR', default='-X-',
		help='sentence boundary')
	arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
		help='character delimiting items in input')
	arg('-o', '--otag', metavar='CHAR', default='O',
		help='alternative outside tag')
	arg('file', nargs='?', default=None)
	return parser.parse_args(argv)

def evaluate_itr(iterable, options=None):
	if options is None:
		options = parse_args([])	# use defaults

	counts = EvalCounts()
	num_features = None		  # number of features per line

	for line in iterable:
		line = line.rstrip('\r\n')

		if options.delimiter == ANY_SPACE:
			features = line.split()
		else:
			features = line.split(options.delimiter)

		if num_features is None:
			num_features = len(features)
		elif num_features != len(features) and len(features) != 0:
			raise FormatError('unexpected number of features: %d (%d)' %
							  (len(features), num_features))

		if len(features) == 0 or features[0] == options.boundary:
			features = [options.boundary, 'O', 'O']
		if len(features) < 3:
			raise FormatError('unexpected number of features in line %s' % line)
		correct, guessed = features[-2], features[-1]

		if correct == guessed:
			counts.correct += 1

		if correct == guessed and correct not in ['-', 'O']:
			counts.t_correct_tags[correct] += 1 #XXX
			counts.correct_tags += 1
		
		if correct not in ['-', 'O']:
			counts.found_correct += 1
			counts.t_found_correct[correct] += 1
		if guessed not in ['-', 'O']:
			counts.found_guessed += 1
			counts.t_found_guessed[guessed] += 1
		counts.num_words += 1

	return counts

def report_notprint(counts, out=None):
	if out is None:
		out = sys.stdout

	overall, by_type = metrics(counts)

	c = counts
	final_report = []
	line = []
	line.append('processed %d eojeols ; ' % (c.num_words))
	line.append('found: %d arguments; correct: %d.\n' % (c.found_guessed, c.correct_tags))
	final_report.append("".join(line))

	if c.num_words > 0:
		line = []
		line.append('accuracy: %6.2f%%; ' % (100.*c.correct/c.num_words))
		line.append('precision: %6.2f%%; ' % (100.*overall.prec))
		line.append('recall: %6.2f%%; ' % (100.*overall.rec))
		line.append('FB1: %6.2f\n' % (100.*overall.fscore))
		final_report.append("".join(line))

	for i, m in sorted(by_type.items()):
		line = []
		line.append('%17s: ' % i)
		line.append('precision: %6.2f%%; ' % (100.*m.prec))
		line.append('recall: %6.2f%%; ' % (100.*m.rec))
		line.append('FB1: %6.2f  %d\n' % (100.*m.fscore, c.t_found_guessed[i]))
		final_report.append("".join(line))
	return final_report

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

def return_report(input_file):
	with codecs.open(input_file, "r", "utf8") as f:
		counts = evaluate_itr(f)
	return report_notprint(counts)

def main(argv):
	args = parse_args(argv[1:])

	if args.file is None:
		counts = evaluate_itr(sys.stdin, args)
	else:
		with open(args.file, encoding='utf-8') as f:
			counts = evaluate_itr(f, args)

if __name__ == '__main__':
	sys.exit(main(sys.argv))
