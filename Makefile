# Annotate data file with ontology terms, run as: make annotate input_file="data/input/TEST/demo_data.xlsx"
annotate:
	@echo "** Annotate data file with ontology terms using config and input_file: $(input_file)"
	@cmd="python src/harmonize.py annotate \
		--config config/config.yml \
		--input_file $(input_file)"; \
	if [ -n "$(output_dir)" ]; then cmd="$$cmd --output_dir $(output_dir)"; fi; \
	if [ -n "$(refresh)" ]; then cmd="$$cmd --refresh"; fi; \
	if [ -n "$(use_openai)" ]; then cmd="$$cmd --use_openai"; fi; \
	echo $$cmd; \
	eval $$cmd
