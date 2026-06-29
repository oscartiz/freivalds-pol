.PHONY: install test figures lint experiments all clean

install:
	pip install -e ".[dev,viz]"

test:
	pytest -q

lint:
	ruff check src tests experiments

figures:
	python -m experiments.figures

experiments:
	@for e in run_detection fp_crux adaptive real_step compressed multiround \
	          curvature_attack backdoor backdoor_capacity; do \
		echo "=== $$e ==="; python -m experiments.$$e || exit 1; \
	done

all: test lint figures

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info
