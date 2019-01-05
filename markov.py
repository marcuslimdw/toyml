from warnings import warn

import numpy as np

import re
import string

import json

import os.path

from functools import reduce

class MCGBase:
	'''Base class for Markov Chain Generators.

	Subclasses should implement _fit methods.'''

	_NORMALISERS = {'simple': lambda x: x,
					'softmax': np.exp}

	def __init__(self, sep=None):
		self.corpus = {}
		self.probabilities = {}
		self.sep = sep

	def _add(self, key, sub_key):

		# A functional solution is actually possible here, but I
		# think it would be inefficient (and not as clean).

		try:
			self.corpus[key][sub_key] += 1

		except KeyError:

			# The sub-key does not exist for this key yet.

			try:
				self.corpus[key][sub_key] = 1

			except KeyError:

				# The key doesn't exist either.

				self.corpus[key] = {sub_key: 1}

	def _sample(self, key):
		if not self.probabilities:
			warn('Probabilities dictionary is empty. Calling fit with default arguments.', RuntimeWarning)
			self.normalise()

		# Transpose the sub_key probabilities dictionary into two tuples.

		sub_keys, probabilities = zip(*self.probabilities[key].items())

		return np.random.choice(sub_keys, p=probabilities)

	def _join(self, strings):
		return self.sep.join(strings)

	def _generate(self, seed, length):

		# Is it possible to write this in a functional style?

		result = [seed]
		for i in range(length):
			try:
				result.append(self._sample(result[-1]))

			except KeyError:
				break
 
		return self._join(result)

	def fit_preprocessed(self, corpus, warm_start=False, auto_normalise=True):
		if not warm_start:
			self.corpus = {}

		for key, sub_key in corpus.items():
			self._add(key, sub_key)

		return self
	
	def fit(self, X, warm_start=True, auto_normalise=True):
		if not warm_start:
			self.corpus = {}

		self._fit(X)
		if auto_normalise:
			self.normalise()

		return self

	def normalise(self, 
			  method='simple', 
			  temperature=1.0):

		if isinstance(method, str):
			normaliser = self._NORMALISERS[method]

		# Get the total number of times each key is a starting key.

		key_sums = {key: sum(map(normaliser, key_pairs.values()))
					for key, key_pairs in self.corpus.items()}

		# Calculate the probability of each sub-key appearing after 
		# each starting key, based on a specified method.

		self.probabilities = {key: {sub_key: normaliser(count) / key_sums[key] 
									for sub_key, count in key_pairs.items()} 
							  for key, key_pairs in self.corpus.items()}

		return self

	def generate(self, seed, length):
		return self._generate(seed, length)

	def save(self, path, mode='error'):
		if mode not in ('error', 'overwrite'):
			raise RuntimeError("Mode accepts 'error' or 'overwrite'; {} was passed".format(mode))

		if mode == 'error' and os.path.isfile(path):
			raise RuntimeError("The file {} already exists. Set mode to 'overwrite' if you wish to overwrite it.")

		else:
			save_data = json.dumps([self.corpus, self.probabilities, self.sep])
			with open(path, 'w') as f:
				f.write(save_data)

		return self

	def load(self, path):
		with open(path, 'r') as f:
			load_data = json.loads(f.read())
		self.corpus, self.probabilities, self.sep = load_data
		return self


class RegexMCG(MCGBase):
	'''Simple Markov Chain Generator '''
	def __init__(self, regex, sep):
		super().__init__(sep)
		self.regex = re.compile(regex)

	def _fit(self, X):
		for document in X:
			doc_words = re.findall(self.regex, document)
			for first, second in zip(doc_words, doc_words[1:]):
				self._add(first, second)

		return self


class WordMCG(RegexMCG):

	_SEPARATE_PUNCTUATION = {True:  '\w+|[{}]'.format(string.punctuation),
							 False: '\w+[{}]?'.format(string.punctuation)}

	def _join(self, strings):
		if self.separate_punctuation:
			def custom_joiner(first, second):
				if second in string.punctuation:
					return ''.join((first, second))

				else:
					return ' '.join((first, second))

			return reduce(custom_joiner, strings)

		else:
			return super()._join(strings)

	def __init__(self, separate_punctuation=True):
		self.separate_punctuation = separate_punctuation
		regex = self._SEPARATE_PUNCTUATION[separate_punctuation]
		super().__init__(regex=regex, sep=' ')


class CharMCG(RegexMCG):
	def __init__(self):
		super().__init__(regex='.', sep = '')

test_corpus = ['I have an apple',
			   'and you have two',
			   'you give me one apple',
			   'and I have two',
			   'I throw away one apple',
			   'and give one back to you',
			   'I don\'t have any apples',
			   'and you have two']