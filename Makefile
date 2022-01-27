.PHONY: \
	black \
	clean \
	clean-build \
	clean-pyc \
	clean-test \
	coverage \
	dist \
	docs \
	flake8 \
	format \
	help \
	install \
	isort \
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

black:  ## run black formatter on code
	black pymedio
	black tests

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

coverage: ## check code coverage quickly with the default Python
	coverage run --source pymedio -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/pymedio.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pymedio
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

flake8: ## check style with flake8
	flake8 pymedio tests

format: black isort flake8 mypy security ## run various code formatters/checks

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

install: clean ## install the package to the active Python's site-packages
	python setup.py install

isort:  ## format imports with isort
	isort pymedio
	isort tests

mypy:  ## typecheck code with mypy
	mypy pymedio
	mypy tests

release: dist ## package and upload a release
	twine upload dist/*

security:  ## run various security checks on code
	bandit -r pymedio -c pyproject.toml
	bandit -r tests -c pyproject.toml

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

snyk:  ## run snyk on to check requirements
	snyk test --file=requirements_dev.txt --package-manager=pip --fail-on=all

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox
