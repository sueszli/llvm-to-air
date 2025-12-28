.PHONY: run
run:
	uv run src/gen_input.py > src/input.ll
	uv run src/air_forge.py src/input.ll > src/test.ll
	xcrun -sdk macosx metal -c src/test.ll -o src/test.air
	xcrun -sdk macosx metallib src/test.air -o src/test.metallib
	clang++ -std=c++17 -framework Metal -framework Foundation src/verify.mm -o src/verify
	./src/verify
	rm -rf src/test.ll src/test.air src/test.metallib src/verify

.PHONY: venv
venv:
	test -f requirements.txt || (uvx pipreqs . --mode no-pin --encoding utf-8 --ignore .venv && mv requirements.txt requirements.in && uv pip compile requirements.in -o requirements.txt)
	uv venv .venv --python 3.11
	uv pip install -r requirements.txt
	@echo "activate venv with: \033[1;33msource .venv/bin/activate\033[0m"

.PHONY: fmt
fmt:
# 	uvx --from cmakelang cmake-format --dangle-parens --line-width 500 -i CMakeLists.txt
	find . -name "*.c" -o -name "*.h" | xargs clang-format -i

	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .
