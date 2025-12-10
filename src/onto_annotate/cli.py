#!/usr/bin/env python3

import warnings

# Suppress pkg_resources deprecation warnings from eutils/setuptools
# Must be set before any imports that trigger eutils
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import click
from datetime import datetime
import uuid
import logging
from oaklib import get_adapter
from oaklib.datamodels.search import SearchProperty, SearchConfiguration
from oaklib.implementations.sqldb.sql_implementation import SqlImplementation
import pandas as pd
import numpy as np
from pathlib import Path
from importlib.resources import files, as_file
from typing import Optional, Dict
from tqdm import tqdm
import sys
import yaml
from pathlib import Path
import os
from openai import OpenAI
import json
import re
import io
from contextlib import contextmanager
import requests
from onto_annotate.entity_type_helper import suggest_entity_types_for_multiple, suggest_entity_types


DEMO_PREFIX = "demo:"
DEMO_BASE = "demo_data"

# BioPortal search cache: (term, ontology_acronyms_tuple, api_key) -> result
_bioportal_cache: Dict[tuple, Optional[Dict]] = {}


__all__ = [
    "main",
]


# Explicitly remove handlers that may have been added automatically
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logger
logger = logging.getLogger("cli")

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s.%(msecs)03d [%(levelname)s] (%(module)s) (%(name)s): %(message)s",
#     datefmt='%Y-%m-%d,%H:%M:%S',
#     handlers=[
#         logging.FileHandler("error.log"),
#         logging.StreamHandler()
#     ],
#     force=True
# )

# Silence root logger
logging.getLogger().setLevel(logging.WARNING)

# Silence oak logger
logging.getLogger("sql_implementation").setLevel(logging.WARNING)


# Silence SQLAlchemy and related sub-loggers
for name in [
    "sqlalchemy",
    "sqlalchemy.engine",
    "sqlalchemy.engine.Engine",
    "sqlalchemy.pool"
]:
    logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger(name).propagate = False


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
# @click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """
    The main click group containing with parameters for logging levels.
    :param verbose: Levels of log messages to display.
    :param quiet: Boolean to be quiet or verbose.
    """
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)


def resolve_input_path(input_arg: str) -> Path:
    """
    Returns a real filesystem Path for both normal files and packaged demo files.
    - "path/to/file.tsv"
    - "demo:<name>" the packaged sample from src/onto_annotate/data/demo_data/<name>
    """
    if input_arg.startswith(DEMO_PREFIX):
        name = input_arg[len(DEMO_PREFIX):].lstrip("/")
        resource = files("onto_annotate").joinpath(f"{DEMO_BASE}/{name}")
        return as_file(resource).__enter__()
    return Path(input_arg).expanduser().resolve()


def resolve_output_dir(output_dir: Optional[str], config_data: Optional[Dict]) -> Path:
    """
    Resolve output directory with priority: CLI -> config -> ./output.
    Ensures the directory exists.
    """
    default = Path.cwd() / "output"
    chosen = output_dir or (config_data or {}).get("output_dir") or default
    out = Path(chosen).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextmanager
def open_input(input_arg: str, encoding: str = "utf-8"):
    """
    Yields a readable text file object for both normal files and packaged demos.

    Usage:
        with open_input(arg) as f:
            df = pd.read_csv(f, sep="\t")
    """
    if isinstance(input_arg, str) and input_arg.startswith(DEMO_PREFIX):
        name = input_arg[len(DEMO_PREFIX):].lstrip("/")
        resource = files("onto_annotate").joinpath(f"{DEMO_BASE}/{name}")
        # Open the resource as bytes, wrap with TextIO so pandas sees text
        with resource.open("rb") as bio:
            with io.TextIOWrapper(bio, encoding=encoding, newline="") as tio:
                yield tio
    else:
        # Normal filesystem path
        with open(Path(input_arg).expanduser().resolve(), "r", encoding=encoding, newline="") as f:
            yield f



def clear_cached_db(ontology_id: str):
    """Clear ontology database files (.db and .db.gz) cached by OAK."""
    base_path = Path.home() / ".data" / "oaklib"
    db_files = [
        base_path / f"{ontology_id}.db",
        base_path / f"{ontology_id}.db.gz"
    ]

    for file in db_files:
        logger.debug("DB refresh set to True.")
        if file.exists():
            logger.debug(f"Removing cached DB: {file}")
            file.unlink()
        else:
            logger.debug(f"No cached DB found at: {file}")




def fetch_ontology(ontology_id: str, refresh: bool = False) -> SqlImplementation:
    """
    Download ontology of interest and convert to SQLite database.
    :param ontology_id: The OBO identifier of the ontology.
    :param refresh: Whether to force refresh the cached ontology.
    :returns adapter: The connector to the ontology database.
    """

    if refresh:
        clear_cached_db(ontology_id)

    # Download and cache the ontology if not already present
    adapter = get_adapter(f"sqlite:obo:{ontology_id}")

    # Display ontology metadata
    try:
        for ont in adapter.ontologies():
            metadata = adapter.ontology_metadata_map(ont)
            version_iri = metadata.get("owl:versionIRI", "unknown")
            logger.debug(f"{ontology_id.upper()} version: {version_iri}")
    except Exception as e:
        logger.debug(f"Could not fetch version metadata for {ontology_id}: {e}")

    return adapter



def load_config(config_path):
    """
    Load YAML config from either a normal path or a packaged demo (demo:config.yml).
    """
    try:
        with open_input(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        click.echo(f"Error reading config file: {e}", err=True)
        sys.exit(1)

    # Basic validation
    required_keys = ["ontologies", "columns_to_annotate"]
    for key in required_keys:
        if key not in config:
            click.echo(f"Missing required config key: {key}", err=True)
            sys.exit(1)

    if not isinstance(config["ontologies"], list) or not all(isinstance(o, str) for o in config["ontologies"]):
        click.echo("Config error: 'ontologies' should be a list of strings", err=True)
        sys.exit(1)

    if not isinstance(config["columns_to_annotate"], list) or not all(isinstance(c, str) for c in config["columns_to_annotate"]):
        click.echo("Config error: 'columns_to_annotate' should be a list of strings", err=True)
        sys.exit(1)

    # Validate BioPortal config if present
    if "bioportal" in config:
        bioportal = config["bioportal"]
        if not isinstance(bioportal, dict):
            click.echo("Config error: 'bioportal' should be a dictionary", err=True)
            sys.exit(1)
        
        if "enabled" in bioportal and not isinstance(bioportal["enabled"], bool):
            click.echo("Config error: 'bioportal.enabled' should be a boolean", err=True)
            sys.exit(1)
        
        if "ontologies" in bioportal:
            if not isinstance(bioportal["ontologies"], list) or not all(isinstance(o, str) for o in bioportal["ontologies"]):
                click.echo("Config error: 'bioportal.ontologies' should be a list of strings", err=True)
                sys.exit(1)

    # Validate entity_type_detector config if present
    if "entity_type_detector" in config:
        etd = config["entity_type_detector"]
        if not isinstance(etd, dict):
            click.echo("Config error: 'entity_type_detector' should be a dictionary", err=True)
            sys.exit(1)
        
        if "enabled" in etd and not isinstance(etd["enabled"], bool):
            click.echo("Config error: 'entity_type_detector.enabled' should be a boolean", err=True)
            sys.exit(1)
        
        if "entity_types" in etd:
            if not isinstance(etd["entity_types"], dict):
                click.echo("Config error: 'entity_type_detector.entity_types' should be a dictionary", err=True)
                sys.exit(1)
            
            # Validate that entity_types values are lists of strings
            for ont_id, types_list in etd["entity_types"].items():
                if not isinstance(types_list, list) or not all(isinstance(t, str) for t in types_list):
                    click.echo(f"Config error: 'entity_type_detector.entity_types.{ont_id}' should be a list of strings", err=True)
                    sys.exit(1)

    return config


def search_ontology(ontology_id: str, adapter: SqlImplementation, df: pd.DataFrame, columns: list, config: dict, desc: str = None) -> pd.DataFrame:
    """
    Search for exact matches to the ontology term label or synonym.
    Supports both single-property searches (LABEL or ALIAS) and combined searches (both).
    :param adapter: The connector to the ontology database.
    :param df: Dataframe containing terms to search and find matches to the ontology.
    :param config: SearchConfiguration with properties to search (can be LABEL, ALIAS, or both).
    :param desc: Optional custom description for progress bar.
    """

    ontology_prefix = 'hpo' if ontology_id.lower() == 'hp' else ontology_id
    exact_search_results = []

    column_to_use = columns[0]  # assuming just one for now

    # Check if this is a combined search (both LABEL and ALIAS)
    properties = [str(p) if hasattr(p, 'name') else str(p) for p in config.properties]
    is_combined_search = 'LABEL' in properties and 'ALIAS' in properties

    # Create a tqdm instance to display search progress
    progress_desc = desc if desc else f"OAK {ontology_id} search"
    progress_bar = tqdm(total=len(df), desc=progress_desc,
                        bar_format='{desc}: [{bar}] {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                        ascii=' #',
                        ncols=80)

    for index, row in df.iterrows():
        search_term = row[column_to_use]
        if pd.isna(search_term):
            progress_bar.update(1)
            continue
            
        norm_term = str(search_term).strip().casefold()
        candidates = []
        
        for result in adapter.basic_search(search_term, config=config):
            label = adapter.label(result)
            
            # Determine match type
            match_type = None
            if is_combined_search:
                # For combined search, check if match is via label or synonym
                if label and str(label).strip().casefold() == norm_term:
                    match_type = "exact_label"
                else:
                    # Check if it matches any EXACT synonym (oboInOwl:exactSynonym only)
                    # We need to verify the synonym type, not just that it's an alias
                    try:
                        exact_synonym_match = False
                        
                        # Try to get exact synonyms from entity metadata
                        # Exact synonyms are stored under predicates like 'oboInOwl:hasExactSynonym'
                        if hasattr(adapter, 'entity_metadata_map'):
                            try:
                                metadata = adapter.entity_metadata_map(result)
                                # Look for exact synonym predicates
                                # OAK uses 'oio:hasExactSynonym' (oboInOwl namespace)
                                exact_synonym_keys = [
                                    'oio:hasExactSynonym',  # OAK's standard key
                                    'oboInOwl:hasExactSynonym',
                                    'http://www.geneontology.org/formats/oboInOwl#hasExactSynonym',
                                    'hasExactSynonym'
                                ]
                                
                                exact_synonyms = []
                                for key in exact_synonym_keys:
                                    if key in metadata:
                                        values = metadata[key]
                                        if isinstance(values, list):
                                            exact_synonyms.extend([str(v) for v in values])
                                        else:
                                            exact_synonyms.append(str(values))
                                
                                # Check if search term matches any exact synonym
                                if exact_synonyms and any(str(s).strip().casefold() == norm_term for s in exact_synonyms):
                                    exact_synonym_match = True
                                
                                # If not found in exact synonyms, check abbreviations/acronyms
                                # (abbreviations/acronyms that are exact matches should also be allowed)
                                if not exact_synonym_match:
                                    abbrev_keys = [
                                        'oio:hasAbbreviation',
                                        'oboInOwl:hasAbbreviation',
                                        'oio:hasAcronym',
                                        'oboInOwl:hasAcronym',
                                        'hasAbbreviation',
                                        'hasAcronym'
                                    ]
                                    
                                    abbrevs = []
                                    for key in abbrev_keys:
                                        if key in metadata:
                                            values = metadata[key]
                                            if isinstance(values, list):
                                                abbrevs.extend([str(v) for v in values])
                                            else:
                                                abbrevs.append(str(values))
                                    
                                    # Only match if it's an exact match (case-insensitive)
                                    # This ensures we don't match partial or broad synonyms
                                    if abbrevs and any(str(a).strip().casefold() == norm_term for a in abbrevs):
                                        exact_synonym_match = True
                            except Exception as e:
                                logger.debug(f"Could not get metadata for {result}: {e}")
                        
                        # If metadata approach didn't work, try direct method (if available)
                        if not exact_synonym_match:
                            try:
                                # Some OAK adapters might have a direct method
                                if hasattr(adapter, 'exact_synonyms'):
                                    exact_synonyms = list(adapter.exact_synonyms(result))
                                    if exact_synonyms and any(str(s).strip().casefold() == norm_term for s in exact_synonyms):
                                        exact_synonym_match = True
                            except Exception:
                                pass
                        
                        # Only set match_type if we verified it's an exact synonym
                        # If we can't verify, skip this match (be conservative)
                        if exact_synonym_match:
                            match_type = "exact_synonym"
                        # else: match_type remains None, so this candidate won't be added
                    except Exception as e:
                        logger.debug(f"Error checking exact synonyms for {result}: {e}")
                        # If we can't verify it's an exact synonym, skip this match
                        match_type = None
                
                # Only add if we determined a match type
                if match_type:
                    candidates.append({
                        "curie": result,
                        "label": label,
                        "match_type": match_type
                    })
            else:
                # Single property search - determine type from config
                if 'LABEL' in properties:
                    match_type = "exact_label"
                elif 'ALIAS' in properties:
                    match_type = "exact_synonym"
                
                candidates.append({
                    "curie": result,
                    "label": label,
                    "match_type": match_type
                })
        
        # For combined search, prioritize label matches over synonym matches, but keep ALL matches
        if is_combined_search and candidates:
            candidates.sort(key=lambda c: 0 if c["match_type"] == "exact_label" else 1)
            # Add all candidates (sorted with label matches first)
            for candidate in candidates:
                exact_search_results.append([row["UUID"], candidate["curie"], candidate["label"], candidate["match_type"]])
        elif candidates:
            # For single property search, take all candidates
            for candidate in candidates:
                exact_search_results.append([row["UUID"], candidate["curie"], candidate["label"], candidate["match_type"]])
        
        # Update the progress bar after processing each row
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Convert search results to dataframe
    results_df = pd.DataFrame(exact_search_results)
    logger.debug(results_df.head())

    # Add column headers
    if results_df.empty:
        results_df = pd.DataFrame(columns=['UUID', f'{ontology_prefix}_result_curie', f'{ontology_prefix}_result_label', f'{ontology_prefix}_result_match_type'])
    else:
        results_df.columns = ['UUID', f'{ontology_prefix}_result_curie', f'{ontology_prefix}_result_label', f'{ontology_prefix}_result_match_type']

    # Filter rows to keep those where '{ontology}_result_curie' starts with the "ontology_id", keep in mind hp vs. hpo
    # TODO: Decide whether these results should still be filtered out
    results_df = results_df[results_df[f'{ontology_prefix}_result_curie'].str.startswith(f'{ontology_id}'.upper())]

    # Group by 'UUID' and aggregate curie and label into lists
    search_results_df = results_df.groupby('UUID').agg({
        f'{ontology_prefix}_result_curie': list,
        f'{ontology_prefix}_result_label': list,
        f'{ontology_prefix}_result_match_type': list
    }).reset_index()

    # Convert lists to strings (join with commas to preserve all matches)
    search_results_df[f'{ontology_prefix}_result_curie'] = search_results_df[f'{ontology_prefix}_result_curie'].apply(lambda x: ', '.join(str(v) for v in x) if x else '')
    search_results_df[f'{ontology_prefix}_result_label'] = search_results_df[f'{ontology_prefix}_result_label'].apply(lambda x: ', '.join(str(v) for v in x) if x else '')
    # For match_type, if there are multiple, prefer label over synonym, otherwise take first
    search_results_df[f'{ontology_prefix}_result_match_type'] = search_results_df[f'{ontology_prefix}_result_match_type'].apply(
        lambda x: 'exact_label' if 'exact_label' in x else (x[0] if x else '')
    )

    return search_results_df



def clean_json_response(content: str) -> str:
    """
    Clean common formatting issues in GPT output before JSON parsing.
    """
    # Replace smart quotes with standard quotes
    content = content.replace("\u201C", "\"").replace("\u201D", "\"").replace("\u2018", "'").replace("\u2019", "'")

    # Strip any leading junk like ```json ... ```
    content = re.sub(r"^```(json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())

    return content


def normalize_bioportal_curie(bioportal_id: str) -> str:
    """
    Normalize BioPortal CURIE/URI to OAK format (PREFIX:ID).
    
    BioPortal may return:
    - Full URI: http://purl.obolibrary.org/obo/MONDO_0006664
    - CURIE: MONDO:0006664
    - Other formats
    
    Returns normalized CURIE in format PREFIX:ID
    """
    if not bioportal_id:
        return ""
    
    # If already in CURIE format (PREFIX:ID), return as-is
    if re.match(r'^[A-Za-z0-9_]+:[A-Za-z0-9_]+$', bioportal_id):
        return bioportal_id
    
    # Try to extract from URI format
    # Pattern: http://purl.obolibrary.org/obo/PREFIX_ID
    obo_match = re.search(r'/obo/([A-Za-z0-9_]+)_([A-Za-z0-9_]+)$', bioportal_id)
    if obo_match:
        prefix = obo_match.group(1)
        identifier = obo_match.group(2)
        return f"{prefix}:{identifier}"
    
    # Try other common URI patterns
    # Pattern: http://.../PREFIX/ID or .../PREFIX#ID
    uri_match = re.search(r'/([A-Za-z0-9_]+)[/#]([A-Za-z0-9_]+)$', bioportal_id)
    if uri_match:
        prefix = uri_match.group(1)
        identifier = uri_match.group(2)
        return f"{prefix}:{identifier}"
    
    # If no pattern matches, try to extract last part as ID
    # and use a generic prefix
    parts = bioportal_id.split('/')
    if len(parts) > 1:
        last_part = parts[-1]
        # Try to split on underscore or colon
        if '_' in last_part:
            prefix, identifier = last_part.split('_', 1)
            return f"{prefix}:{identifier}"
        elif ':' in last_part:
            return last_part
    
    # Fallback: return as-is if we can't normalize
    logger.debug(f"Could not normalize BioPortal CURIE: {bioportal_id}")
    return bioportal_id


def search_bioportal(term: str, ontology_acronyms: tuple, api_key: str) -> Optional[Dict]:
    """
    Single BioPortal call across all ontologies; client-side exact matching and priority selection.
    Returns first match by priority order; prefers exact label over synonym when sorting.
    
    :param term: Search term
    :param ontology_acronyms: Tuple of ontology acronyms to search (in priority order)
    :param api_key: BioPortal API key
    :return: Dict with 'curie', 'label', 'ontology_acronym', 'match_type' if match found, None otherwise
    """
    if not api_key:
        return None
    
    if not ontology_acronyms:
        return None
    
    # Check cache
    cache_key = (term, ontology_acronyms, api_key)
    if cache_key in _bioportal_cache:
        return _bioportal_cache[cache_key]
    
    # Limit cache size to 1000 entries
    if len(_bioportal_cache) >= 1000:
        # Clear oldest 100 entries (simple FIFO)
        keys_to_remove = list(_bioportal_cache.keys())[:100]
        for key in keys_to_remove:
            del _bioportal_cache[key]
    
    try:
        url = "https://data.bioontology.org/search"
        params = {
            "q": term,
            "ontologies": ",".join(ontology_acronyms),
            "apikey": api_key,
            "pagesize": 50,
            "include": "prefLabel,synonym",
            "also_search_properties": "true",
            "require_exact_match": "false",
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        norm_term = term.strip().casefold()
        candidates = []
        
        if "collection" in data and len(data["collection"]) > 0:
            for result in data["collection"]:
                label = result.get("prefLabel", "") or result.get("label", "")
                synonyms = result.get("synonym", []) or []
                
                # Determine ontology acronym from response (try ontologyAcronym, fallback to link)
                ontology_acronym = result.get("ontologyAcronym", "")
                if not ontology_acronym:
                    ontology_link = result.get("links", {}).get("ontology", "")
                    ontology_acronym = ontology_link.split("/")[-1] if ontology_link else ""
                ontology_acronym = (ontology_acronym or "").upper()
                
                curie_id = result.get("@id", "")
                normalized_curie = normalize_bioportal_curie(curie_id)
                if not normalized_curie or not ontology_acronym:
                    continue
                
                match_type = None
                if label and label.strip().casefold() == norm_term:
                    match_type = "exact_label"
                elif any(isinstance(s, str) and s.strip().casefold() == norm_term for s in synonyms):
                    match_type = "exact_synonym"
                else:
                    continue
                
                candidates.append({
                    "curie": normalized_curie,
                    "label": label,
                    "ontology_acronym": ontology_acronym,
                    "match_type": match_type
                })
        
        if not candidates:
            _bioportal_cache[cache_key] = None
            return None
        
        # Priority sort by config order; prefer label over synonym
        priority = {ont.upper(): i for i, ont in enumerate(ontology_acronyms)}
        candidates.sort(key=lambda c: (
            priority.get(c["ontology_acronym"], 1e9),
            0 if c["match_type"] == "exact_label" else 1
        ))
        
        result_dict = candidates[0]
        _bioportal_cache[cache_key] = result_dict
        return result_dict
    
    except requests.exceptions.RequestException as e:
        logger.debug(f"BioPortal API error (all ontologies): {e}")
        return None
    except Exception as e:
        logger.debug(f"Error processing BioPortal result (all ontologies): {e}")
        return None
    
    # Cache None result to avoid repeated failed searches
    _bioportal_cache[cache_key] = None
    return None


def get_alternative_names(term: str) -> dict:
    """Create a prompt for ChatGPT to ask for alternative names."""
    prompt = (
        f"For the term '{term}', return a JSON object with three keys:\n"
        "- 'input': the original term\n"
        "- 'alt_names': a list of up to five alternative names\n"
        "- 'source': the string 'openai'\n\n"
        "Example format:\n"
        "{\n"
        '  "input": "hole in heart",\n'
        '  "alt_names": [\n'
        '    "atrial septal defect",\n'
        '    "ventricular septal defect",\n'
        '    "congenital heart defect",\n'
        '    "septal defect",\n'
        '    "cardiac shunt"\n'
        "  ],\n"
        '  "source": "openai"\n'
        "}"
    )

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
            temperature=0.5,
            max_output_tokens=150,
        )
        content = response.output_text
        cleaned = clean_json_response(content)
        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError:
        logger.debug(f"Could not parse JSON: {content}")
        return []
    except Exception as e:
        logger.debug(f"Error retrieving alt names for '{term}': {e}")
        return []


def detect_entities_by_type(text: str, entity_types: list[str]) -> list[dict]:
    """
    Use LLM to detect entities of specific types in text.
    
    Args:
        text: Text to analyze
        entity_types: List of entity types to search for (e.g., ["disease", "condition"])
        
    Returns:
        List of dictionaries with 'span' and 'label' keys for each detected entity
    """
    if not entity_types:
        return []
    
    # Create prompt for entity detection
    entity_types_str = ", ".join(entity_types)
    prompt = (
        f"Given the following text, are there any entities of type {entity_types_str}? "
        f"Return a JSON object with a list of entities found. "
        f"Each entity should have 'span' (the exact text span from the input) and 'label' (the entity name).\n\n"
        f"Text: {text}\n\n"
        f"Return format:\n"
        "{\n"
        '  "entities": [\n'
        '    {"span": "text span", "label": "entity name"},\n'
        '    {"span": "another span", "label": "another entity"}\n'
        "  ]\n"
        "}"
    )
    
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
            temperature=0.3,
            max_output_tokens=300,
        )
        content = response.output_text
        cleaned = clean_json_response(content)
        result = json.loads(cleaned)
        
        # Extract entities list
        entities = result.get("entities", [])
        if not isinstance(entities, list):
            return []
        
        # Validate and return entities
        valid_entities = []
        for entity in entities:
            if isinstance(entity, dict) and "span" in entity and "label" in entity:
                valid_entities.append({
                    "span": str(entity["span"]),
                    "label": str(entity["label"])
                })
        
        return valid_entities
        
    except json.JSONDecodeError as e:
        logger.debug(f"Could not parse JSON from entity detection: {content if 'content' in locals() else 'N/A'}")
        return []
    except Exception as e:
        logger.debug(f"Error detecting entities in text: {e}")
        return []


def generate_uuid() -> str:
    """Function to generate UUID"""
    return str(uuid.uuid4())


def _clean_up_columns(df: pd.DataFrame, ontology_id: str) -> pd.DataFrame:
    """
    Copy over the search results to the columns from the input dataframe
    amd remove the extra columns added with the search results.
    :param df: The dataframe from the merge of the search results with the original dataframe.
    :param ontology_id: The ontology identifier, ie. the ontology being searched  
    """
    ontology_prefix = 'hpo' if ontology_id.lower() == 'hp' else ontology_id

    # Handle clean-up after a second round of synonym search
    if f'{ontology_prefix}_result_match_type_x' in df.columns and f'{ontology_prefix}_result_match_type_y' in df.columns:
        # Copy result label values to original df column and then drop result column
        df[f'{ontology_prefix}Label'] = np.where(df[f'{ontology_prefix}_result_label'].notnull(), df[f'{ontology_prefix}_result_label'], df[f'{ontology_prefix}Label'])
        df.drop([f'{ontology_prefix}_result_label'], axis=1, inplace=True)

        # Copy result curie values to original df column and then drop result column
        df[f'{ontology_prefix}Code'] = np.where(df[f'{ontology_prefix}_result_curie'].notnull(), df[f'{ontology_prefix}_result_curie'], df[f'{ontology_prefix}Code'])
        df.drop([f'{ontology_prefix}_result_curie'], axis=1, inplace=True)

        # Copy type of result match to original df column and then drop result column and rename original df column
        df[f'{ontology_prefix}_result_match_type_x'] = np.where(df[f'{ontology_prefix}_result_match_type_y'].notnull(), df[f'{ontology_prefix}_result_match_type_y'], df[f'{ontology_prefix}_result_match_type_x'])
        df.drop([f'{ontology_prefix}_result_match_type_y'], axis=1, inplace=True)
        df = df.rename(columns={f'{ontology_prefix}_result_match_type_x': f'{ontology_prefix}_result_match_type'})
    else:
        # Update values in the existing columns
        df[f'{ontology_prefix}Label'] = df[f'{ontology_prefix}_result_label']
        df[f'{ontology_prefix}Code'] = df[f'{ontology_prefix}_result_curie']
    
        # Drop the search_results columns
        df.drop([f'{ontology_prefix}_result_label'], axis=1, inplace=True)
        df.drop([f'{ontology_prefix}_result_curie'], axis=1, inplace=True)


    return df


def _check_ontology_versions(ontology_ids: tuple):
    """
    Check and print the cached ontology versions for each ID.
    :param ontology_ids: Tuple of ontology IDs to check.
    """
    logger.debug("\nChecking local ontology versions...")
    for oid in ontology_ids:
        try:
            adapter = get_adapter(f"sqlite:obo:{oid}")
            for ont in adapter.ontologies():
                meta = adapter.ontology_metadata_map(ont)
                version = meta.get("owl:versionIRI", "Unknown")
                logger.debug(f"  - {oid.upper()}: {version}")
        except Exception as e:
            logger.debug(f"  - {oid.upper()}: Error loading version ({e})")



def custom_join(series):
    non_empty = series.dropna().astype(str).str.strip()
    non_empty = [v for v in non_empty if v]
    unique_vals = list(dict.fromkeys(non_empty))
    return ', '.join(unique_vals) if unique_vals else ''


@main.command("suggest-entity-types")
@click.option('--ontology', multiple=True, required=True, help='Ontology ID (e.g., MONDO, HP). Can be specified multiple times.')
@click.option('--output', type=click.Path(), required=False, help='Optional output file path (JSON or YAML). If not specified, prints to stdout.')
def suggest_entity_types_cmd(ontology, output):
    """
    Suggest entity types for ontologies based on OBO Foundry metadata.
    """
    from onto_annotate.entity_type_helper import suggest_entity_types_for_multiple
    
    ontology_list = list(ontology)
    click.echo(f"Fetching entity type suggestions for: {', '.join(ontology_list)}")
    
    results = suggest_entity_types_for_multiple(ontology_list)
    
    # Format output
    if output:
        output_path = Path(output)
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, sort_keys=True)
            click.echo(f"Entity type suggestions written to: {output_path}")
        else:
            # Default to JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Entity type suggestions written to: {output_path}")
    else:
        # Print to stdout
        click.echo("\nSuggested entity types:")
        click.echo("=" * 50)
        for ont_id, entity_types in sorted(results.items()):
            if entity_types:
                click.echo(f"\n{ont_id}:")
                for et in entity_types:
                    click.echo(f"  - {et}")
            else:
                click.echo(f"\n{ont_id}: (no suggestions available)")


@main.command("annotate")
@click.option('--config', type=click.Path(exists=True), help='Path to YAML config file')
@click.option('--input_file', type=str, required=True, help="Path to data file to annotate (or 'demo:<name>.tsv')")
@click.option('--output_dir', type=click.Path(), required=False, help='Optional override for output directory (default: ./output)')
@click.option('--refresh', is_flag=True, help='Force refresh of ontology cache')
@click.option('--no_openai', is_flag=True, help='Disable OpenAI-based fallback searches')
def annotate(config: str, input_file: str, output_dir: str, refresh: bool, no_openai: bool):
    """
    Annotate a data file with ontology terms.
    :param config: Path to the config file.
    :param input_file: The name of the file with terms to search for ontology matches.
    """

    config_data = load_config(config)

    ontologies = config_data["ontologies"]
    columns = config_data["columns_to_annotate"]

    if not ontologies or not columns:
        raise click.ClickException("Config file must contain 'ontologies' and 'columns_to_annotate'.")
    
    # Determine output directory from CLI or config or fallback
    output_dir = resolve_output_dir(output_dir, config_data)

    click.echo(f"Using ontologies: {ontologies}")
    click.echo(f"Annotating columns: {columns}")
    click.echo(f"Output directory: {output_dir}")
    if refresh:
        click.echo("Refresh mode enabled")

    oid = tuple(o.lower() for o in config_data["ontologies"])
 
    filename_prefix = '_'.join(oid)

    # Get the current formatted timestamp
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y%m%d-%H%M%S")

    # Read in the data file
    input_path = resolve_input_path(input_file)

    # data_df = pd.read_csv(input_path, sep='\t')
    with open_input(input_file) as fh:
        data_df = pd.read_csv(fh, sep="\t")
    logger.debug(data_df[columns])
    
    # Add a new column 'UUID' with unique identifier values
    data_df['UUID'] = data_df.apply(lambda row: generate_uuid(), axis=1)
    logger.debug(data_df.nunique())
    logger.info("Number of total rows in dataframe: %s", len(data_df))

    # Check ontology versions 
    _check_ontology_versions(oid)


    # Collect all annotation results
    all_final_results = []

    for ontology_id in oid:
        ontology_prefix = 'hpo' if ontology_id.lower() == 'hp' else ontology_id
        adapter = fetch_ontology(ontology_id, refresh=refresh)

        # === Combined OAK search (label and synonym) ===
        combined_config = SearchConfiguration(properties=[SearchProperty.LABEL, SearchProperty.ALIAS], force_case_insensitive=True)
        combined_hits_df = search_ontology(ontology_id, adapter, data_df, columns, combined_config)
        
        # Split results by match type
        label_hits_df = combined_hits_df[combined_hits_df[f'{ontology_prefix}_result_match_type'] == 'exact_label'].copy()
        synonym_hits_df = combined_hits_df[combined_hits_df[f'{ontology_prefix}_result_match_type'] == 'exact_synonym'].copy()
        
        # Remove synonym matches that already have label matches (prioritize labels)
        matched_uuids_label = set(label_hits_df["UUID"])
        synonym_hits_df = synonym_hits_df[~synonym_hits_df["UUID"].isin(matched_uuids_label)]
        
        # Tag results
        label_hits_df["annotation_source"] = "oak"
        label_hits_df["annotation_method"] = "exact_label"
        label_hits_df["ontology"] = ontology_id.lower()

        synonym_hits_df["annotation_source"] = "oak"
        synonym_hits_df["annotation_method"] = "exact_synonym"
        synonym_hits_df["ontology"] = ontology_id.lower()

        # Filter unmatched for next stage
        matched_uuids_all = set(combined_hits_df["UUID"])
        filtered_df = data_df[~data_df["UUID"].isin(matched_uuids_all)]

        # === OpenAI alternative names ===
        openai_hits = []
        if not no_openai:
            # Step 1: Collect all alternative terms from OpenAI
            all_alt_terms_data = []  # List of dicts: {uuid, original_term, original_row_term, alt_term, alt_names}
            
            for term in tqdm(filtered_df[columns[0]].dropna().unique(), desc="OpenAI search",
                            bar_format='{desc}: [{bar}] {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                            ascii=' #',
                            ncols=80):
                # Normalize the term
                match = re.match(r"^(.*?)\s*\((\w{3,5})\)$", term.strip())
                normalized_term = match.group(1).strip() if match else term.strip()

                alt_response = get_alternative_names(normalized_term)
                if not alt_response:
                    continue

                uuid_series = filtered_df[filtered_df[columns[0]] == term]["UUID"]
                if uuid_series.empty:
                    continue
                uuid = uuid_series.iloc[0]

                # Collect all alternative terms for batch search
                for alt in alt_response.get("alt_names", []):
                    all_alt_terms_data.append({
                        "uuid": uuid,
                        "original_term": normalized_term,
                        "original_row_term": term,
                        "alt_term": alt,
                        "alt_names": alt_response.get("alt_names", [])
                    })
            
            # Step 2: Batch search all alternative terms at once
            if all_alt_terms_data:
                # Print status message for curator visibility
                num_alt_terms = len(all_alt_terms_data)
                num_original_terms = len(set(item["uuid"] for item in all_alt_terms_data))
                print(f"\n🔍 Searching {num_alt_terms} LLM-generated alternative terms (from {num_original_terms} original terms) against OAK {ontology_id.upper()}...")
                
                # Create dataframe for batch OAK search
                alt_search_df = pd.DataFrame([
                    {"UUID": item["uuid"], columns[0]: item["alt_term"]}
                    for item in all_alt_terms_data
                ])
                
                # Single OAK search for all alternative terms with custom description
                alt_matches_df = search_ontology(
                    ontology_id, 
                    adapter, 
                    alt_search_df, 
                    columns, 
                    combined_config,
                    desc=f"OAK {ontology_id} (LLM alt terms)"
                )
                
                # Step 3: Process matches and map back to original terms
                # Track which original UUIDs found matches (only first match per UUID)
                matched_original_uuids = set()
                
                if not alt_matches_df.empty:
                    # Process matches, keeping only first match per original UUID
                    for _, match_row in alt_matches_df.iterrows():
                        match_uuid = match_row["UUID"]
                        
                        # Skip if we already have a match for this original UUID
                        if match_uuid in matched_original_uuids:
                            continue
                        
                        # Find the original data for this UUID
                        # Since all alt_terms for a UUID come from the same original term,
                        # we can use any of them to get the original_data
                        original_data = None
                        for item in all_alt_terms_data:
                            if item["uuid"] == match_uuid:
                                original_data = item
                                break

                        if original_data:
                            # Tag the match with OpenAI metadata
                            # Convert Series to dict for safe modification
                            match_dict = match_row.to_dict()
                            match_type = match_dict.get(f'{ontology_prefix}_result_match_type', '')
                            
                            if match_type == 'exact_label':
                                match_dict["annotation_source"] = "openai"
                                match_dict["annotation_method"] = "alt_term_label"
                            else:
                                match_dict["annotation_source"] = "openai"
                                match_dict["annotation_method"] = "alt_term_synonym"
                            match_dict["original_term"] = original_data["original_term"]
                            match_dict["ontology"] = ontology_id.lower()
                            
                            # Create a single-row dataframe for this match
                            match_df = pd.DataFrame([match_dict])
                            openai_hits.append(match_df)
                            matched_original_uuids.add(match_uuid)
                
                # Step 4: Track terms that didn't match (for BioPortal)
                # Create records for original terms that had no matches
                processed_no_match_uuids = set()
                for item in all_alt_terms_data:
                    if item["uuid"] not in matched_original_uuids and item["uuid"] not in processed_no_match_uuids:
                        openai_hits.append(pd.DataFrame([{
                            "UUID": item["uuid"],
                        "annotation_source": "openai",
                        "annotation_method": "no_match",
                            "original_term": item["original_term"],
                            "alt_names": ', '.join(item["alt_names"]),
                        "ontology": ontology_id.lower()
                        }]))
                        processed_no_match_uuids.add(item["uuid"])


        openai_hits_df = pd.concat(openai_hits, ignore_index=True) if openai_hits else pd.DataFrame()
        
        # Track all matched UUIDs so far (OAK + OpenAI, excluding no_match records)
        matched_uuids_all = set(combined_hits_df["UUID"])
        if not openai_hits_df.empty:
            # Only count UUIDs that actually matched (not no_match records)
            matched_uuids_openai = set(
                openai_hits_df[openai_hits_df["annotation_method"] != "no_match"]["UUID"].dropna()
            )
            matched_uuids_all |= matched_uuids_openai
        
        # === LLM Entity-Type Detector ===
        entity_detector_hits = []
        etd_config = config_data.get("entity_type_detector", {})
        if etd_config.get("enabled", False) and not no_openai:
            # Get entity types for this ontology
            entity_types = None
            if "entity_types" in etd_config:
                # User-defined entity types
                entity_types = etd_config["entity_types"].get(ontology_id.upper(), [])
            
            # If not user-defined, try to derive from OBO Foundry
            if not entity_types:
                entity_types = suggest_entity_types(ontology_id)
            
            if entity_types:
                # Get unmatched texts after OAK + OpenAI
                unmatched_df = data_df[~data_df["UUID"].isin(matched_uuids_all)]
                
                if not unmatched_df.empty:
                    # Collect all entity detection data
                    all_entity_data = []  # List of dicts: {uuid, original_text, entity_span, entity_label, entity_type}
                    
                    for _, row in tqdm(unmatched_df.iterrows(), total=len(unmatched_df), 
                                      desc=f"Entity type detection ({ontology_id})",
                                      bar_format='{desc}: [{bar}] {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                                      ascii=' #',
                                      ncols=80):
                        text = row[columns[0]]
                        if pd.isna(text):
                            continue
                        
                        # Normalize the text (same as OpenAI step)
                        match = re.match(r"^(.*?)\s*\((\w{3,5})\)$", str(text).strip())
                        normalized_text = match.group(1).strip() if match else str(text).strip()
                        
                        # Detect entities
                        detected_entities = detect_entities_by_type(normalized_text, entity_types)
                        
                        if detected_entities:
                            uuid = row["UUID"]
                            for entity in detected_entities:
                                # Determine which entity type this matches (if multiple types provided)
                                entity_type_used = None
                                entity_label_lower = entity["label"].lower()
                                for et in entity_types:
                                    if et.lower() in entity_label_lower or entity_label_lower in et.lower():
                                        entity_type_used = et
                                        break
                                if not entity_type_used and entity_types:
                                    entity_type_used = entity_types[0]  # Default to first type
                                
                                all_entity_data.append({
                                    "uuid": uuid,
                                    "original_text": normalized_text,
                                    "entity_span": entity["span"],
                                    "entity_label": entity["label"],
                                    "entity_type": entity_type_used
                                })
                    
                    # Batch search all detected entities through OAK
                    if all_entity_data:
                        num_entities = len(all_entity_data)
                        num_original_texts = len(set(item["uuid"] for item in all_entity_data))
                        print(f"\n🔍 Searching {num_entities} LLM-detected entities (from {num_original_texts} texts) against OAK {ontology_id.upper()}...")
                        
                        # Create dataframe for batch OAK search
                        entity_search_df = pd.DataFrame([
                            {"UUID": item["uuid"], columns[0]: item["entity_label"]}
                            for item in all_entity_data
                        ])
                        
                        # Single OAK search for all detected entities
                        entity_matches_df = search_ontology(
                            ontology_id,
                            adapter,
                            entity_search_df,
                            columns,
                            combined_config,
                            desc=f"OAK {ontology_id} (entity detection)"
                        )
                        
                        # Process matches and map back to original texts
                        matched_entity_uuids = set()
                        
                        if not entity_matches_df.empty:
                            for _, match_row in entity_matches_df.iterrows():
                                match_uuid = match_row["UUID"]
                                
                                # Skip if we already have a match for this UUID
                                if match_uuid in matched_entity_uuids:
                                    continue
                                
                                # Find the original entity data
                                original_entity_data = None
                                for item in all_entity_data:
                                    if item["uuid"] == match_uuid:
                                        original_entity_data = item
                                        break
                                
                                if original_entity_data:
                                    # Tag the match
                                    match_dict = match_row.to_dict()
                                    match_type = match_dict.get(f'{ontology_prefix}_result_match_type', '')
                                    
                                    match_dict["annotation_source"] = "llm_entity_detector"
                                    match_dict["annotation_method"] = "entity_type_extraction"
                                    match_dict["entity_type"] = original_entity_data.get("entity_type", "")
                                    match_dict["original_text"] = original_entity_data["original_text"]
                                    match_dict["detected_span"] = original_entity_data["entity_span"]
                                    match_dict["ontology"] = ontology_id.lower()
                                    
                                    # Create a single-row dataframe for this match
                                    match_df = pd.DataFrame([match_dict])
                                    entity_detector_hits.append(match_df)
                                    matched_entity_uuids.add(match_uuid)
                        
                        # Update matched UUIDs
                        matched_uuids_all |= matched_entity_uuids
        
        entity_detector_hits_df = pd.concat(entity_detector_hits, ignore_index=True) if entity_detector_hits else pd.DataFrame()
        
        # Update matched UUIDs after entity detection
        if not entity_detector_hits_df.empty:
            matched_uuids_entity = set(entity_detector_hits_df["UUID"].dropna())
            matched_uuids_all |= matched_uuids_entity
        
        # === BioPortal fallback search ===
        bioportal_hits = []
        bioportal_config = config_data.get("bioportal", {})
        if bioportal_config.get("enabled", False):
            # Get API key from environment variable
            bioportal_api_key = os.getenv("BIOPORTAL_API_KEY", "")
            bioportal_ontologies = bioportal_config.get("ontologies", [])
            
            if bioportal_api_key and bioportal_ontologies:
                # Get terms that still have no match (after OAK and OpenAI)
                unmatched_df = data_df[~data_df["UUID"].isin(matched_uuids_all)]
                
                if not unmatched_df.empty:
                    for bp_term in tqdm(unmatched_df[columns[0]].dropna().unique(), desc="BioPortal search",
                                        bar_format='{desc}: [{bar}] {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                                        ascii=' #',
                                        ncols=80):
                        # Normalize the term (same as OpenAI)
                        match = re.match(r"^(.*?)\s*\((\w{3,5})\)$", bp_term.strip())
                        normalized_term = match.group(1).strip() if match else bp_term.strip()
                        
                        bioportal_result = search_bioportal(
                            normalized_term,
                            tuple(bioportal_ontologies),
                            bioportal_api_key
                        )
                        
                        if bioportal_result:
                            # Get all UUIDs for this term (may appear in multiple rows)
                            uuid_series = unmatched_df[unmatched_df[columns[0]] == bp_term]["UUID"]
                            if uuid_series.empty:
                                continue
                            
                            bp_ontology = bioportal_result["ontology_acronym"].lower()
                            match_type_label = "EXACT_LABEL" if bioportal_result["match_type"] == "exact_label" else "EXACT_SYNONYM"
                            
                            # Create a hit for each UUID that has this term
                            for uuid in uuid_series:
                                bioportal_hits.append(pd.DataFrame([{
                                    "UUID": uuid,
                                    "bioportal_result_curie": bioportal_result["curie"],
                                    "bioportal_result_label": bioportal_result["label"],
                                    "bioportal_result_match_type": match_type_label,
                                    "annotation_source": "bioportal",
                                    "annotation_method": "exact_label" if bioportal_result["match_type"] == "exact_label" else "exact_synonym",
                                    "ontology": bp_ontology
                                }]))
            elif bioportal_config.get("enabled", False):
                logger.warning("BioPortal is enabled but API key or ontologies list is missing. Skipping BioPortal search.")
        
        bioportal_hits_df = pd.concat(bioportal_hits, ignore_index=True) if bioportal_hits else pd.DataFrame()
        
        # Keep only the last search method's results (even if no_match)
        # If BioPortal is enabled, it's the last search, so remove all OpenAI no_match rows
        if bioportal_config.get("enabled", False) and not openai_hits_df.empty:
            # Remove all OpenAI no_match rows since BioPortal is the last search method
            openai_hits_df = openai_hits_df[openai_hits_df["annotation_method"] != "no_match"]
        
        results_sources = [label_hits_df, synonym_hits_df]
        if not openai_hits_df.empty:
            results_sources.append(openai_hits_df)
        if not entity_detector_hits_df.empty:
            results_sources.append(entity_detector_hits_df)
        if not bioportal_hits_df.empty:
            results_sources.append(bioportal_hits_df)

        results_df = pd.concat(results_sources, ignore_index=True)
        if not results_df.empty:
            all_final_results.append(results_df)

    # === Exit if no results  ===
    if not all_final_results:
        logger.warning("No annotation results found for any ontology.")
        # Always write a TSV with expected columns even when there are no matches
        final_df = data_df.copy()

        # Ensure string type (avoid NaNs becoming 'nan' later)
        for c in final_df.columns:
            final_df[c] = final_df[c].astype(str)

        # Add empty result columns for each requested ontology
        for oid_single in oid:  # e.g., ('mondo',) or ('mondo','hp')
            prefix = 'hpo' if oid_single.lower() == 'hp' else oid_single.lower()
            for col in (
                f"{prefix}_result_curie",
                f"{prefix}_result_label",
                f"{prefix}_result_match_type",
            ):
                if col not in final_df.columns:
                    final_df[col] = ""

        final_df = final_df.fillna("")

        output_path = Path(output_dir) / f"{filename_prefix}-combined_ontology_annotations-{formatted_timestamp}.tsv"
        final_df.to_csv(output_path, sep="\t", index=False)

        print(f"\nAnnotation results written to: {output_path}")
        return


    # === Combine all results ===
    full_results = pd.concat(all_final_results, ignore_index=True)

    # === Merge with input metadata ===
    data_df["UUID"] = data_df["UUID"].astype(str).str.strip()
    full_results["UUID"] = full_results["UUID"].astype(str).str.strip()
    final_df = pd.merge(data_df, full_results, on="UUID", how="left")

    # Drop duplicated or unnecessary columns
    final_df.drop(columns=["source_column_value_y", "original_term"], inplace=True, errors="ignore")

    # === Save final output ===
    final_df = final_df.fillna("")
    output_path = Path(output_dir) / f"{filename_prefix}-combined_ontology_annotations-{formatted_timestamp}.tsv"
    final_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nAnnotation results written to: {output_path}")


if __name__ == "__main__":
    main()