# OntoAnnotate
Annotate data to ontology terms. This script takes in a TSV file with short text snippets and annotates the text with ontology terms. 
The text can be a few words to a paragraph (see example input files in `data/demo_data`), additional preprocessing of the text is currently needed before using this tool to chunk the text.

The current version of the script assumes the ontologies to use for annotation exist as a [SemSQL](https://github.com/incatools/semantic-sql) database. Future versions will remove this requirement.

## Requirements
Python

## Create and activate a virtual environment
The prerequisites to run the script can either be installed using the `requirements.txt` file if using pyenv or `environment.yml` file if using conda.

### Commands
`pip install -r requirements.txt`
Followed by: 
`python -m venv .venv && source .venv/bin/activate`

** OR **

`conda env create -f environment.yml`
Followed by:
`conda activate onto-annotate`

## Install in editable mode
To make the `onto-annotate` command available in your environment from the root of the project directory run:
`pip install -e ".[dev]"`

Use of the `-e` flag means that any code changes to files in `src/onto_annotate` will take effect immediately without reinstalling.

## Configuration
Copy the example config and customize it for your project:
`cp config/config.example.yml config/config.yml`

The YAML file has three keys: `ontologies`, `column_to_annotate`, and `output_dir`. The key `ontologies` should contain the ontology "acronym" as listed on [BioPortal](https://bioportal.bioontology.org/ontologies). The `column_to_annotate` key is the column header name in the input data file that contains the text to annotate. Finally, there is an optional key, `output_dir` if a location other than `data/output` is preferred for the resulting annotated files.


## OpenAI API
The tool has an option to use the OpenAI API to annotate text not otherwise matched to an ontology term. In order to use this feature, create an OpenAI API Key [here](https://platform.openai.com/api-keys) and then add this your environment as:
`export OPENAI_API_KEY=<YOUR-API-KEY>`


## Usage
Annotate your text files (without AI assistance) as:
```
onto-annotate annotate \
  --config config/config.yml \
  --input_file demo:conditions_simple.tsv \
  --output_dir data/output \
  --no_openai  # Remove this flag to annotate with AI assistance
```

NOTES:
1. `--no_openai` - this is a boolean flag to skip LLM-based annotation. To use OpenAI simply do not add this flag and remember to set your OPENAI_API_KEY as described above.
1. `--refresh` flag to update the cached OAK ontology database. To rely on the existing local copy, leave out `--refresh` or `refresh=true`.
1. `--output_dir` is optional; it can be defined in the YAML config instead.
1. Use `-vv` before `annotate` to generate debug output.


## Data File
The script reads and writes TSV files. The prefixes of the ontologies to be used for the annotation can be added into the config.yml file.

### Input file
See example input file `conditions_simple.tsv` in `data/demo_data`.

### Output file
Example output file: 

```
condition_source_text	UUID	mondo_result_curie	mondo_result_label	mondo_result_match_type	annotation_source	annotation_method	ontology	alt_names	hpo_result_curie	hpo_result_label	hpo_result_match_type	maxo_result_curie	maxo_result_label	maxo_result_match_type
ASD	7317c559-ff88-4c31-8608-77615b20b267	MONDO:0006664	atrial septal defect	MONDO_EXACT_ALIAS	oak	exact_synonym	mondo							
ASD	7317c559-ff88-4c31-8608-77615b20b267				oak	exact_synonym	hp		HP:0000729, HP:0001631	Autistic behavior, Atrial septal defect	HPO_EXACT_ALIAS			
ASD	7317c559-ff88-4c31-8608-77615b20b267				openai	no_match	maxo	autism spectrum disorder, atrial septal defect, advanced sleep phase disorder, asynchronous serial data, active server directory						
```

## Run Tests
Pytest is used as the testing framework and all tests can be run (in quiet mode) as: `pytest -q`


### Optional Make Command
`make annotate input_file=toy_data/raw_data_conditions/conditions_simple.tsv output_dir=harmonica/tmp/output refresh=true`
(the `make` command is run from the root of the project)

Optional parameters of `refresh=true` and `use_openai=true` can be added.
TODO: Update based on new repo structure.


## Ontology SQLite Database
Using `get_adapter(f"sqlite:obo:{ontology_id}")` the ontology database is saved at `~/.data/oaklib/`.

NOTE: This method downloads a version of an ontolgy from an AWS S3 bucket (https://s3.amazonaws.com/bbop-sqlite) managed by the OAK developers (https://github.com/INCATools/ontology-access-kit). Only one version of an ontology is present in the S3 bucket.

Since OAK does not have a mechanism to automatically update the local cached ontology database (saved to `~/.data/oaklib/`), a custom method was added to harmonica. This gets the release date from the cached ontology database(s) and displays these to the user with a prompt asking whether to use these cached versions or download updated versions, where these updated versions are the latest version/content that is in the AWS S3 bucket. After downloading the latest content from the S3 bucket, the ontology release date is displayed again to the user and then the annotation process occurs.

There is a cache control option for OAK, however this manages the default cache expiry lifetime of 7 days. This does not ensure that when the data annotation is run that it's using the latest ontology content available. As of this code update (31-Mar-2025), the `refresh` option is only available in the OAK commandline and not in the Python code.

OAK references:
- Cached ontology database is out of date - https://incatools.github.io/ontology-access-kit/faq/troubleshooting.html#my-cached-sqlite-ontology-is-out-of-date

- Cache control - https://incatools.github.io/ontology-access-kit/cli.html#cache-control


TODO: Include other methods to download ontology content and convert to a SQLite database using [semsql](https://github.com/INCATools/semantic-sql) or add additional step to query ontologies in [BioPortal](https://bioportal.bioontology.org/). Note, BioPortal does support having private ontologies.

