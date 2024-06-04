NOTEBOOKS = $(wildcard *.ipynb)

all: $(NOTEBOOKS)
	@for notebook in $(NOTEBOOKS); do \
		echo "Compiling $$notebook"; \
		pytest --nbmake $$notebook; \
		jupyter nbconvert --to pdf $$notebook; \
	done

.PHONY: all