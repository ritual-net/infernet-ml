.phony: clean build-library publish-to-local-pypi start-local-pypi

clean:
	rm -rf dist
	rm -rf src/infernet_ml.egg-info

build-library: clean
	# build library from pyproject.toml
	python -m build

publish-to-local-pypi: build-library
	twine upload --repository-url http://localhost:4040 dist/* --username user --password user

start-local-pypi:
	mkdir -p packages
	pypi-server run -a . -P . -p 4040 packages --overwrite
