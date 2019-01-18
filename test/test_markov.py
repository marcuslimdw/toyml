import pytest

from numpy import exp

from toyml.markov import *

@pytest.fixture
def simple_mcg():
	mcg = MCGBase()
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
	mcg = MCGBase()
	self.assertEqual(mcg.corpus, {})
	mcg._add('key', 'value_one')
	self.assertEqual(mcg.corpus, {'key': {'value_one': 1}})
	mcg._add('key', 'value_two')
	self.assertEqual(mcg.corpus, {'key': {'value_one': 1,
										  'value_two': 1}})
	mcg._add('key', 'value_one')
	self.assertEqual(mcg.corpus, {'key': {'value_one': 2,
										  'value_two': 1}})

def test_normalise_simple(simple_mcg):
	simple_mcg.normalise(method='simple')
	self.assertEqual(mcg.probabilities, {'key_one': {'value_one': 1 / 6,
												    'value_two': 2 / 6,
													'value_three': 3 / 6},
										 'key_two': {'value_one': 3 / 8,
										    	     'value_two': 5 / 8}})

	self.assertTrue(all(1 - sum(key_pairs.values()) < 0.0001
						for key_pairs in mcg.probabilities.values()))

def test_normalise_softmax():
	mcg.normalise(method='softmax')
	key_one_sum = sum(map(exp, mcg.corpus['key_one'].values()))
	key_two_sum = sum(map(exp, mcg.corpus['key_two'].values()))
	self.assertEqual(mcg.probabilities, {'key_one': {'value_one': exp(1) / key_one_sum,
												    'value_two': exp(2) / key_one_sum,
													'value_three': exp(3) / key_one_sum},
										 'key_two': {'value_one': exp(3) / key_two_sum,
										    	     'value_two': exp(5) / key_two_sum}})

	self.assertTrue(all(1 - sum(key_pairs.values()) < 0.0001
						for key_pairs in mcg.probabilities.values()))

def test_normalise_simple_temperature():
	mcg.normalise(method='simple', temperature=2.0)
	self.assertEqual(mcg.probabilities, {'key_one': {'value_one': 1 / 6,
												    'value_two': 2 / 6,
													'value_three': 3 / 6},
										 'key_two': {'value_one': 3 / 8,
										    	     'value_two': 5 / 8}})

	self.assertTrue(all(1 - sum(key_pairs.values()) < 0.0001
						for key_pairs in mcg.probabilities.values()))

def test_normalise_softmax_temperature():
	mcg.normalise(method='softmax', temperature=2.0)
	key_one_sum = sum(map(lambda x: exp(x / 2), mcg.corpus['key_one'].values()))
	key_two_sum = sum(map(lambda x: exp(x / 2), mcg.corpus['key_two'].values()))
	self.assertEqual(mcg.probabilities, {'key_one': {'value_one': exp(1 / 2) / key_one_sum,
												    'value_two': exp(2 / 2) / key_one_sum,
													'value_three': exp(3 / 2) / key_one_sum},
										 'key_two': {'value_one': exp(3 / 2) / key_two_sum,
										    	     'value_two': exp(5 / 2) / key_two_sum}})

	self.assertTrue(all(1 - sum(key_pairs.values()) < 0.0001
						for key_pairs in mcg.probabilities.values()))

def test_sample():
	pass

def test_sample_warn():
	with self.assertWarns(RuntimeWarning):
	mcg._sample('key_one')

def test_save():
	save_path = 'test_save.json'
	self.assertFalse(os.path.isfile(save_path))
	mcg.save(save_path)
	self.assertTrue(os.path.isfile(save_path))
	os.remove(save_path)

def test_save_error():
	save_path = 'test_save_error.json'
	with open(save_path, 'w') as f:
	pass

	with self.assertRaises(RuntimeError):
	mcg.save(save_path)

	self.assertEqual(os.path.getsize(save_path), 0)

	os.remove(save_path)		

def test_save_overwrite():
	save_path = 'test_save_overwrite.json'
	with open(save_path, 'w') as f:
	pass
		
	mcg.save(save_path, 'overwrite')
	self.assertNotEqual(os.path.getsize(save_path), 0)

	os.remove(save_path)

def test_load():
	mcg = MCGBase()
	load_path = 'test_load.json'
	self.assertFalse(os.path.isfile(load_path))
	with open(load_path, 'w') as f:
	f.write(json.dumps([{}, {}, '']))

	mcg.load(load_path)

	os.remove(load_path)

def test_save_load():
	mcg_original = self.make_mcg()
	mcg_original.normalise()
	mcg_data = MCGBase()
	save_load_path = 'test_save_load.json'
	self.assertFalse(os.path.isfile(save_load_path))

	mcg_original.save(save_load_path)
	mcg_data.load(save_load_path)

	self.assertEqual(mcg_original.corpus, mcg_data.corpus)
	self.assertEqual(mcg_original.probabilities, mcg_data.probabilities)
	self.assertEqual(mcg_original.sep, mcg_data.sep)

	os.remove(save_load_path)		

def test_join():
	mcg = WordMCG()
	expected = 'I don\'t have any apples, and you have two.'
	result = mcg._join(['I', 'don', "'", 't', 'have', 'any', 'apples', ',', 'and', 'you', 'have', 'two', '.'])
	self.assertEqual(expected, result)

def test_fit_separate_punctuation():
	mcg = WordMCG()
	self.assertEqual(mcg.corpus, {})
	mcg.fit(self.corpus)
	self.assertEqual(mcg.corpus, {'word_one': {'word_two': 2},
								  'word_two': {',': 2,
								  			   'word_two': 1,
								  			   'word_four': 1},
								  ',':		  {'word_three': 1,
								  			   'word_four': 1}})

def test_fit_not_separate_punctuation():
	mcg = WordMCG(separate_punctuation=False)
	self.assertEqual(mcg.corpus, {})
	mcg.fit(self.corpus)
	self.assertEqual(mcg.corpus, {'word_one':  {'word_two,':  2},
								  'word_two':  {'word_two':   1,
								  				'word_four':  1},
								  'word_two,': {'word_three': 1,
									  		    'word_four':  1}})
