import numpy as np

_METRICS = {'manhattan': lambda p1, p2: sum(abs(p1 - p2)),
		    'euclidean': lambda p1, p2: sum((p1 - p2) ** 2) ** 0.5,
		    'chebyshev': lambda p1, p2: max(abs(p1 - p2))}

def check_array_like(*args):
	return map(np.array, args)

def almost_equal(lhs, rhs):
	pass
	
def normalise(x):
	return x / sum(x)

def identity(x):
	return x

def softmax(x):
	return normalise(np.exp(x))

def dist(p1, 
		 p2, 
		 metric='euclidean'):
			   
	return _METRICS[metric](p1, p2)
