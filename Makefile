.PHONY: test
test:
	uv run --with pytest --with pyobjc-framework-Metal --with pyobjc-framework-Cocoa pytest test/

.PHONY: venv
venv:
	test -f requirements.txt || (uvx pipreqs . --mode no-pin --encoding utf-8 --ignore .venv && mv requirements.txt requirements.in && uv pip compile requirements.in -o requirements.txt)
	uv venv .venv --python 3.11
	uv pip install -r requirements.txt
	@echo "activate venv with: \033[1;33msource .venv/bin/activate\033[0m"

.PHONY: fmt
fmt:
	find . -name "*.c" -o -name "*.h" | xargs clang-format -i
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
