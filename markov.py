import numpy as np

import re
import string

import json
import os.path

from functools import reduce
from operator import itemgetter

from .utils import softmax


class MCGBase:
    '''Base class for Markov Chain Generators.

    Subclasses should implement the _fit method.'''

    def __init__(self, sep=None):
        self.corpus = {}
        self.base_probabilities = {}
        self.sep = sep

    def fit(self, X, warm_start=True):
        if not warm_start:
            self.corpus = {}

        self._fit(X)
        self.normalise()
        return self

    def fit_preprocessed(self, corpus, warm_start=False):
        if not warm_start:
            self.corpus = {}

        for key, sub_key in corpus.items():
            self._add(key, sub_key)

        self.normalise()
        return self

    def normalise(self):
        '''Convert sub-key counts to probabilities.'''

        key_sums = {key: sum(key_pairs.values()) for key, key_pairs in self.corpus.items()}
        self.base_probabilities = {key: {sub_key: count / key_sums[key] for sub_key, count in key_pairs.items()}
                                   for key, key_pairs in self.corpus.items()}

        return self

    def generate(self, seed, length, temperature=1.0):
        return self._generate(seed, length, temperature)

    def save(self, path, mode='error'):
        if mode not in ('error', 'overwrite'):
            raise RuntimeError("Mode accepts 'error' or 'overwrite'; {} was passed".format(mode))

        if mode == 'error' and os.path.isfile(path):
            raise RuntimeError("The file {} already exists. Set mode to 'overwrite' if you wish to overwrite it.")

        else:
            save_data = json.dumps([self.corpus, self.base_probabilities, self.sep])
            with open(path, 'w') as f:
                f.write(save_data)

        return self

    def load(self, path):
        with open(path, 'r') as f:
            load_data = json.loads(f.read())

        self.corpus, self.base_probabilities, self.sep = load_data
        return self

    def _add(self, key, sub_key):

        # A functional solution is actually possible here, but I think it would be inefficient (and not as clean).

        try:
            self.corpus[key][sub_key] += 1

        except KeyError:

            # The sub-key does not exist for this key yet.

            try:
                self.corpus[key][sub_key] = 1

            except KeyError:

                # The key doesn't exist either.

                self.corpus[key] = {sub_key: 1}

    def _sample(self, key, temperature=1.0):
        # Transpose the sub_key probabilities dictionary into two tuples.

        if temperature == 0:
            return max(self.base_probabilities[key].items(), key=itemgetter(1))[0]

        sub_keys, probabilities = zip(*self.base_probabilities[key].items())

        if temperature > 0:
            return np.random.choice(sub_keys, p=softmax(np.array(probabilities) / temperature))

        else:
            return np.random.choice(sub_keys, p=softmax(np.array(probabilities) * temperature))

    def _join(self, strings):
        return self.sep.join(strings)

    def _generate(self, seed, length, temperature=1.0):

        # Is it possible to write this in a functional style?

        result = [seed]
        for i in range(length):
            try:
                result.append(self._sample(result[-1], temperature))

            except KeyError:
                break

        return self._join(result)


class RegexMCG(MCGBase):
    '''Simple Markov Chain generator implementing _fit with regex logic. Finds all non-overlapping instances of a given
    regex.'''
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

    _SEPARATE_PUNCTUATION = {True:  r'\w+|[{}]'.format(string.punctuation),
                             False: r'\w+[{}]?'.format(string.punctuation)}

    def _join(self, strings):
        if self.separate_punctuation:
            def custom_joiner(first, second):
                if second in string.punctuation or first[-1] == "\'":
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
        super().__init__(regex='.', sep='')
