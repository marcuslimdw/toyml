clean:
	find -type f -name "*.pyc" -delete
	find -type d -name __pycache__ -delete

test:
	for filename in ./test/test_*.py; \
	do pytest $$filename; \
	done;

.PHONY: test