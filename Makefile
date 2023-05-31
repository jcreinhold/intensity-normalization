.PHONY: \
	black \
	clean \
	clean-build \
	clean-pyc \
	clean-test \
	clean-test-all \
	coverage \
	develop \
	dist \
	docs \
	format \
	help \
	install \
	isort \
	lint \
	mypy \
	release \
	security \
	servedocs \
	snyk \
	test \
	test-all

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

black:  ## run black on python code
	black intensity_normalization
	black tests

clean: clean-build clean-pyc clean-test  ## remove all build, test, coverage and Python artifacts

clean-build:  ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:  ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:  ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-test-all: clean-test  ## remove all test artifacts
	rm -fr .tox/
	rm -fr .mypy_cache

coverage:  ## check code coverage quickly with the default Python
	coverage run --source intensity_normalization -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

develop: clean  ## symlink the package to the active Python's site-packages
	python setup.py develop

dist: clean  ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

docs:  ## generate Sphinx HTML documentation, including API docs
	rm -f docs/intensity_normalization.rst
	rm -f docs/intensity_normalization.*.rst
	sphinx-apidoc -o docs/ intensity_normalization
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

format: black isort mypy security  ## format and run various checks on the code

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

install: clean  ## install the package to the active Python's site-packages
	python setup.py install

isort:  ## run isort on python code
	isort intensity_normalization
	isort tests

lint:  ## check style with flake8
	flake8 intensity_normalization tests

mypy:  ## typecheck code with mypy
	mypy intensity_normalization
	mypy tests

release: dist  ## package and upload a release
	twine upload dist/*

security:  ## run various code quality checks and formatters
	bandit -r medio -c pyproject.toml
	bandit -r tests -c pyproject.toml

servedocs: docs  ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

snyk:  security  ## run snyk for dependency security checks
	snyk test --file=requirements_dev.txt --package-manager=pip --fail-on=all

test:  ## run tests quickly with the default Python
	pytest --cov=intensity_normalization --disable-pytest-warnings

test-all:  ## run tests on every Python version with tox
	tox
