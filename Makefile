clean:
	find -type f -name "*.pyc" -delete
	find -type d -name __pycache__ -delete

format:
	find ./ -name "*.py" -exec sed -i "s/[ \t]*$$//" {} \;
	find ./ -name "*.py" -exec sed -i -e '$$a\' {} \;

flake8:
	find ./ -name "*.py" -exec flake8 {} +

test: flake8 test-force benchmark

test-fast: flake8 test-fast-force

test-fast-force:
	find ./test -name "*.py" -exec pytest {} -m "not slow" \;

test-force:
	find ./test/ -name "*.py" -exec pytest {} \;

benchmark:
	python3 ./benchmark/benchmark.py

.PHONY: test, benchmark