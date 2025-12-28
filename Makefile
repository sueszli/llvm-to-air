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

# 
# conda
# 

# generate environment.yml from requirements.txt (idempotent)
.PHONY: reqs-to-yaml
reqs-to-yaml:
	conda update -n base -c defaults conda
	conda config --env --set subdir osx-arm64 || true
	conda config --set auto_activate_base false
	conda info
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate base; \
		conda create --yes --name con python=3.11; \
		source $$(conda info --base)/etc/profile.d/conda.sh && conda activate con; \
		pip install -r requirements.txt; \
		conda env export --no-builds | grep -v "prefix:" > environment.yml; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
		conda remove --yes --name con --all; \
	'

# init conda from environment.yml file
.PHONY: conda
conda:
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda env create --file environment.yml --solver=libmamba; \
	'
	@echo "activate conda with: \033[1;33mconda activate con\033[0m"

.PHONY: conda-clean
conda-clean:
	# conda clean --all
	bash -c '\
		source $$(conda info --base)/etc/profile.d/conda.sh; conda activate base; \
		conda remove --yes --name con --all; \
		source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate; \
	'

# 
# utils
# 

.PHONY: fmt
fmt:
	uvx isort .
	uvx autoflake --remove-all-unused-imports --recursive --in-place .
	uvx black --line-length 5000 .

.PHONY: md-to-pdf
md-to-pdf:
	pandoc "$(filepath)" -o "$(basename $(filepath)).pdf"

.PHONY: rmd-to-pdf
rmd-to-pdf:
	Rscript -e 'for(p in c("rmarkdown", "ISLR", "IRkernel")) if(!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cran.rstudio.com")'
	Rscript -e "rmarkdown::render('$(filepath)', output_format = 'pdf_document')"
	rm -rf *.bib *.aux *.log *.out *.synctex.gz