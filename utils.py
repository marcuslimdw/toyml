def _check_array_like(*args):
	return True

def identity(x):
	return x

def dist(p1, 
		  p2, 
		  metric='euclidean'):

	metrics = {'manhattan': lambda p1, p2: abs(p1 - p2),
			   'euclidean': lambda p1, p2: sum((p1 - p2) ** 2) ** 0.5,
			   'chebyshev': lambda p1, p2: max(abs(p1 - p2))}
			   
	return metrics[metric](p1, p2)

