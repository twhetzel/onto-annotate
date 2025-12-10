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

The YAML file has the following keys:
- `ontologies`: List of ontology acronyms (as listed on [BioPortal](https://bioportal.bioontology.org/ontologies)) to search using OAK
- `columns_to_annotate`: List of column header names in the input data file that contain the text to annotate
- `output_dir`: (Optional) Output directory for annotated files (default: `data/output`)
- `entity_type_detector`: (Optional) Dictionary for LLM entity-type detector configuration:
  - `enabled` (boolean): Enable/disable entity-type detection
  - `entity_types` (dict): Mapping of ontology ID to list of entity types to detect (e.g., `MONDO: ["disease", "condition", "disorder"]`)
  - If `entity_types` is not provided, entity types will be auto-derived from OBO Foundry metadata based on the ontology's domain
  
  **Note:** Set `OPENAI_API_KEY` environment variable to use this feature. Run `onto-annotate suggest-entity-types --ontology MONDO` to get suggested entity types.
- `bioportal`: (Optional) Dictionary for BioPortal fallback search configuration:
  - `enabled` (boolean): Enable/disable BioPortal search
  - `ontologies` (list): List of ontology acronyms to search in BioPortal (searched in priority order)
  
  **Note:** Set `BIOPORTAL_API_KEY` environment variable to use this feature


## OpenAI API
The tool uses the OpenAI API for two features:
1. **Alternative name generation**: Generates alternative names for unmatched terms
2. **Entity-type detection**: Extracts entities of specific types from complex text (e.g., questions, paragraphs)

In order to use these features, create an OpenAI API Key [here](https://platform.openai.com/api-keys) and then add it to your environment as:
`export OPENAI_API_KEY=<YOUR-API-KEY>`

### Entity-Type Detector
The entity-type detector extracts entities of specified types from text that didn't match in initial OAK searches. This is particularly useful for complex text like survey questions or paragraphs where entities may be embedded.

**Pipeline order:**
1. OAK exact match search (labels and synonyms)
2. OpenAI alternative name generation
3. LLM entity-type detection** (if enabled)
4. BioPortal fallback search

**Getting suggested entity types:**
Use the helper command to get entity type suggestions based on OBO Foundry metadata:
```bash
onto-annotate suggest-entity-types --ontology MONDO --ontology HP
```

This will output suggested entity types that you can add to your config file:
```yaml
entity_type_detector:
  enabled: true
  entity_types:
    MONDO: ["disease", "condition", "disorder"]
    HP: ["phenotype", "symptom", "clinical feature"]
```

If you don't specify `entity_types` in the config, the tool will automatically derive them from OBO Foundry metadata based on the ontology's domain.

## BioPortal API
The tool has an optional BioPortal fallback search that runs after OAK and OpenAI searches fail. BioPortal searches for exact matches on preferred labels first, then synonyms if no exact match is found. To use this feature:

1. Create a BioPortal account and obtain an API key from [BioPortal](https://bioportal.bioontology.org/)
2. Set the API key as an environment variable: `export BIOPORTAL_API_KEY=<YOUR-API-KEY>`
3. Enable BioPortal in your config file and specify which ontologies to search:
   ```yaml
   bioportal:
     enabled: true
     ontologies:
       - ICD10CM
       - SNOMEDCT
       # Searched in priority order until match found
   ```


## Usage

### Annotate data files
Annotate your text files (without AI assistance) as:
```
onto-annotate annotate \
  --config config/config.yml \
  --input_file demo:conditions_simple.tsv \
  --output_dir data/output \
  --no_openai  # Remove this flag to annotate with AI assistance
```

### Suggest entity types
Get suggested entity types for ontologies based on OBO Foundry metadata:
```
onto-annotate suggest-entity-types \
  --ontology MONDO \
  --ontology HP \
  --output entity_types.yaml  # Optional: save to file
```

**NOTES:**
1. `--no_openai` - this is a boolean flag to skip all LLM-based annotation (alternative names and entity-type detection). To use OpenAI features, simply do not add this flag and remember to set your `OPENAI_API_KEY` as described above.
2. `--refresh` flag to update the cached OAK ontology database. To rely on the existing local copy, leave out `--refresh` or `refresh=true`.
3. `--output_dir` is optional; it can be defined in the YAML config instead.
4. Use `-vv` before `annotate` to generate debug output.


## Data File
The script reads and writes TSV files. The prefixes of the ontologies to be used for the annotation can be added into the config.yml file.

### Input file
See example input file `conditions_simple.tsv` in `data/demo_data`.

### Output file
The output file includes columns for each ontology searched, plus metadata columns:
- `annotation_source`: Source of the match (`oak`, `openai`, `llm_entity_detector`, `bioportal`)
- `annotation_method`: Method used (`exact_label`, `exact_synonym`, `alt_term_label`, `alt_term_synonym`, `entity_type_extraction`, `no_match`)
- `ontology`: Ontology ID (lowercase)
- `entity_type`: (Entity-type detector only) The entity type that was detected
- `original_text`: (Entity-type detector only) The original text from which the entity was extracted
- `detected_span`: (Entity-type detector only) The text span that was detected as an entity
- `alt_names`: (OpenAI only) Alternative names generated by LLM

Example output file: 

```
condition_source_text	UUID	mondo_result_curie	mondo_result_label	mondo_result_match_type	annotation_source	annotation_method	ontology	entity_type	original_text	detected_span
ASD	7317c559-ff88-4c31-8608-77615b20b267	MONDO:0006664	atrial septal defect	exact_synonym	oak	exact_synonym	mondo					
heart defect	49914056-f414-4859-8c68-192845b2487a	MONDO:0005453	congenital heart disease	exact_synonym	llm_entity_detector	entity_type_extraction	mondo	disease	Were any of the following problems... - Complications from heart defect...	heart defect
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

