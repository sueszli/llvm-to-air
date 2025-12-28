# 
# venv
# 

# init venv from imports
.PHONY: venv
venv:
	test -f requirements.txt || (uvx pipreqs . --mode no-pin --encoding utf-8 --ignore .venv && mv requirements.txt requirements.in && uv pip compile requirements.in -o requirements.txt)
	uv venv .venv --python 3.11
	uv pip install -r requirements.txt
	@echo "activate venv with: \033[1;33msource .venv/bin/activate\033[0m"

# dump + compile dependencies
.PHONY: lock
lock:
	uv pip freeze > requirements.in
	uv pip compile requirements.in -o requirements.txt

.PHONY: fmt
fmt:
# 	uvx --from cmakelang cmake-format --dangle-parens --line-width 500 -i CMakeLists.txt
	find . -name "*.c" -o -name "*.h" | xargs clang-format -i

	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .

# 
# docker
# 

DOCKER_RUN = docker run --rm -p 9090:9090 -v $(PWD):/workspace main sh -c

.PHONY: docker-build
docker-build:
	docker build -t main .
 
.PHONY: docker-run
docker-run:
	$(DOCKER_RUN) "python3 /workspace/src/mnist.py"

.PHONY: docker-clean
docker-clean:
	# docker compose down --rmi all --volumes --remove-orphans
	# docker system prune -a -f
	docker rmi -f main:latest
