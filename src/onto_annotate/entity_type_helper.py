"""
Helper module for suggesting entity types based on OBO Foundry metadata.
"""

import json
import logging
import requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

OBO_FOUNDRY_URL = "https://obofoundry.org/registry/ontologies.jsonld"

# Domain to entity type mappings
DOMAIN_TO_ENTITY_TYPES = {
    "health": ["disease", "condition", "disorder", "syndrome"],
    "phenotype": ["phenotype", "symptom", "clinical feature", "trait"],
    "anatomy and development": ["anatomical structure", "body part", "organ"],
    "biological systems": ["biological process", "pathway", "system"],
    "chemistry and biochemistry": ["chemical", "molecule", "compound"],
    "investigations": ["test", "procedure", "examination", "diagnostic test"],
    "organisms": ["organism", "species", "taxon"],
    "environment": ["environmental factor", "exposure"],
    "diet, metabolomics, and nutrition": ["nutrient", "metabolite", "dietary component"],
    "microbiology": ["microorganism", "pathogen", "bacterium", "virus"],
    "information": ["information entity", "data", "record"],
    "information technology": ["software", "algorithm", "computational method"],
    "agriculture": ["crop", "agricultural entity"],
    "simulation": ["simulation", "model"],
    "upper": ["entity", "process", "continuant"],
}


def fetch_obo_foundry_data() -> Optional[Dict]:
    """
    Fetch OBO Foundry registry data.
    
    Returns:
        Dictionary with 'ontologies' list, or None if fetch fails
    """
    try:
        response = requests.get(OBO_FOUNDRY_URL, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch OBO Foundry data: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OBO Foundry JSON: {e}")
        return None


def get_ontology_metadata(ontology_id: str, obo_data: Optional[Dict] = None) -> Optional[Dict]:
    """
    Get metadata for a specific ontology from OBO Foundry.
    
    Args:
        ontology_id: Ontology ID (e.g., "mondo", "hp")
        obo_data: Optional pre-fetched OBO Foundry data
        
    Returns:
        Dictionary with ontology metadata, or None if not found
    """
    if obo_data is None:
        obo_data = fetch_obo_foundry_data()
    
    if not obo_data or "ontologies" not in obo_data:
        return None
    
    ontology_id_lower = ontology_id.lower()
    
    for ontology in obo_data["ontologies"]:
        # Check id, preferredPrefix, and handle special cases like hp/hpo
        ont_id = ontology.get("id", "").lower()
        prefix = ontology.get("preferredPrefix", "").lower()
        
        if (ont_id == ontology_id_lower or 
            prefix == ontology_id_lower or
            (ontology_id_lower == "hp" and ont_id == "hp") or
            (ontology_id_lower == "hpo" and ont_id == "hp")):
            return ontology
    
    return None


def suggest_entity_types(ontology_id: str, obo_data: Optional[Dict] = None) -> List[str]:
    """
    Suggest entity types for an ontology based on OBO Foundry metadata.
    
    Args:
        ontology_id: Ontology ID (e.g., "MONDO", "HP")
        obo_data: Optional pre-fetched OBO Foundry data
        
    Returns:
        List of suggested entity type strings
    """
    metadata = get_ontology_metadata(ontology_id, obo_data)
    
    if not metadata:
        logger.warning(f"Could not find metadata for ontology: {ontology_id}")
        return []
    
    suggested_types = []
    
    # Get domain-based suggestions
    domain = metadata.get("domain", "").lower()
    if domain in DOMAIN_TO_ENTITY_TYPES:
        suggested_types.extend(DOMAIN_TO_ENTITY_TYPES[domain])
    
    # Get tag-based suggestions
    tags = metadata.get("tags", [])
    for tag in tags:
        tag_lower = str(tag).lower()
        # Add tag itself if it's a meaningful entity type
        if tag_lower in ["disease", "phenotype", "symptom", "condition", "disorder", 
                         "syndrome", "trait", "procedure", "test"]:
            if tag_lower not in suggested_types:
                suggested_types.append(tag_lower)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_types = []
    for t in suggested_types:
        if t not in seen:
            seen.add(t)
            unique_types.append(t)
    
    return unique_types


def suggest_entity_types_for_multiple(ontology_ids: List[str]) -> Dict[str, List[str]]:
    """
    Suggest entity types for multiple ontologies.
    
    Args:
        ontology_ids: List of ontology IDs
        
    Returns:
        Dictionary mapping ontology ID to list of suggested entity types
    """
    obo_data = fetch_obo_foundry_data()
    
    results = {}
    for ont_id in ontology_ids:
        results[ont_id.upper()] = suggest_entity_types(ont_id, obo_data)
    
    return results
