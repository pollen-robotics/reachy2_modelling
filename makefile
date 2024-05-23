format:
	python -m black .
	python -m isort --profile black .
	# python -m autoflake --in-place --remove-all-unused-imports --recursive .

coverage:
	coverage run --source=src -m pytest -vv .

test: coverage format
	coverage report
