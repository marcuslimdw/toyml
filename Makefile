clean:
	find -type f -name "*.pyc" -delete
	find -type d -name __pycache__ -delete

test:
	python3 ./test/test_markov.py
	python3 ./test/test_utils.py

.PHONY: test