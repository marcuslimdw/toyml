from toyml import markov

import pytest

import numpy as np

import os

import json


@pytest.fixture
def simple_mcg():
    mcg = markov.MCGBase()
    mcg.corpus = {'key_one': {'value_one': 1,
                              'value_two': 2,
                              'value_three': 3},
                  'key_two': {'value_one': 3,
                              'value_two': 5}}
    return mcg


@pytest.fixture
def corpus():
    return ['word_one word_two, word_three',
            'word_one word_two, word_four',
            'word_two word_two word_four']


def test_add():
    mcg = markov.MCGBase()
    assert mcg.corpus == {}
    mcg._add('key', 'value_one')
    assert mcg.corpus == {'key': {'value_one': 1}}
    mcg._add('key', 'value_two')
    assert mcg.corpus == {'key': {'value_one': 1,
                          'value_two': 1}}
    mcg._add('key', 'value_one')
    assert mcg.corpus == {'key': {'value_one': 2,
                          'value_two': 1}}


def test_normalise(simple_mcg):
    simple_mcg.normalise()
    assert simple_mcg.base_probabilities == {'key_one': {'value_one': 1 / 6,
                                                         'value_two': 2 / 6,
                                                         'value_three': 3 / 6},
                                             'key_two': {'value_one': 3 / 8,
                                                         'value_two': 5 / 8}}

    np.testing.assert_allclose([sum(key_pairs.values())
                                for key_pairs in simple_mcg.base_probabilities.values()], 1.0)


# def test_normalise_softmax(simple_mcg):
#     simple_mcg.normalise(method='softmax')
#     key_one_sum = sum(map(np.exp, simple_mcg.corpus['key_one'].values()))
#     key_two_sum = sum(map(np.exp, simple_mcg.corpus['key_two'].values()))
#     assert simple_mcg.base_probabilities == {'key_one': {'value_one': np.exp(1) / key_one_sum,
#                                                          'value_two': np.exp(2) / key_one_sum,
#                                                         'value_three': np.exp(3) / key_one_sum},
#                                         'key_two': {'value_one': np.exp(3) / key_two_sum,
#                                                     'value_two': np.exp(5) / key_two_sum}}

#     np.testing.assert_allclose(sum(sum(key_pairs.values())
#                                for key_pairs in simple_mcg.base_probabilities.values()), 1.0)


# def test_normalise_simple_temperature(simple_mcg):
#     simple_mcg.normalise(method='simple', temperature=2.0)
#     assert simple_mcg.base_probabilities == {'key_one': {'value_one': 1 / 6,
#                                                          'value_two': 2 / 6,
#                                                          'value_three': 3 / 6},
#                                              'key_two': {'value_one': 3 / 8,
#                                                          'value_two': 5 / 8}}

#     np.testing.assert_allclose((sum(key_pairs.values()) for key_pairs in simple_mcg.base_probabilities.values()), 1.0)


# def test_normalise_softmax_temperature(simple_mcg):
#     simple_mcg.normalise(method='softmax', temperature=2.0)
#     key_one_sum = sum(map(lambda x: np.exp(x / 2), simple_mcg.corpus['key_one'].values()))
#     key_two_sum = sum(map(lambda x: np.exp(x / 2), simple_mcg.corpus['key_two'].values()))
#     assert simple_mcg.base_probabilities == {'key_one': {'value_one': np.exp(1 / 2) / key_one_sum,
#                                                          'value_two': np.exp(2 / 2) / key_one_sum,
#                                                          'value_three': np.exp(3 / 2) / key_one_sum},
#                                              'key_two': {'value_one': np.exp(3 / 2) / key_two_sum,
#                                                          'value_two': np.exp(5 / 2) / key_two_sum}}

#     np.testing.assert_allclose((sum(key_pairs.values()) for key_pairs in simple_mcg.base_probabilities.values()), 1.0)


def test_sample():
    pass


def test_save(simple_mcg):
    save_path = 'test_save.json'
    assert not os.path.isfile(save_path)
    simple_mcg.save(save_path)
    assert os.path.isfile(save_path)
    os.remove(save_path)


def test_save_error(simple_mcg):
    save_path = 'test_save_error.json'
    with open(save_path, 'w'):
        pass

    with pytest.raises(RuntimeError):
        simple_mcg.save(save_path)

    assert os.path.getsize(save_path) == 0

    os.remove(save_path)


def test_save_overwrite(simple_mcg):
    save_path = 'test_save_overwrite.json'
    with open(save_path, 'w') as f:
        f.write('')

    assert os.path.getsize(save_path) == 0
    simple_mcg.save(save_path, 'overwrite')
    assert os.path.getsize(save_path) != 0

    os.remove(save_path)


def test_load():
    mcg = markov.MCGBase()
    load_path = 'test_load.json'
    assert not os.path.isfile(load_path)
    with open(load_path, 'w') as f:
        f.write(json.dumps([{}, {}, '']))

    mcg.load(load_path)
    os.remove(load_path)


def test_save_load(simple_mcg):
    mcg_original = simple_mcg
    mcg_original.normalise()
    mcg_data = markov.MCGBase()
    save_load_path = 'test_save_load.json'
    assert not os.path.isfile(save_load_path)

    mcg_original.save(save_load_path)
    mcg_data.load(save_load_path)

    assert mcg_original.corpus == mcg_data.corpus
    assert mcg_original.base_probabilities == mcg_data.base_probabilities
    assert mcg_original.sep == mcg_data.sep

    os.remove(save_load_path)


def test_join():
    mcg = markov.WordMCG()
    expected = 'I don\'t have any apples, and you have two.'
    result = mcg._join(['I', 'don', "'", 't', 'have', 'any', 'apples', ',', 'and', 'you', 'have', 'two', '.'])
    assert expected == result


def test_fit_separate_punctuation(corpus):
    mcg = markov.WordMCG()
    assert mcg.corpus == {}
    mcg.fit(corpus)
    assert mcg.corpus == {'word_one': {'word_two': 2},
                          'word_two': {',': 2,
                                       'word_two': 1,
                                       'word_four': 1},
                          ','       : {'word_three': 1,  # noqa: E203
                                       'word_four': 1}}


def test_fit_not_separate_punctuation(corpus):
    mcg = markov.WordMCG(separate_punctuation=False)
    assert mcg.corpus == {}
    mcg.fit(corpus)
    assert mcg.corpus == {'word_one':  {'word_two,':  2},
                          'word_two':  {'word_two':   1,
                                        'word_four':  1},
                          'word_two,': {'word_three': 1,
                                        'word_four':  1}}
