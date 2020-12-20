# STM-corpus with coreference resolution and knowledge graph population

This repository contains the annotated STM corpus with coreferences, populated knowledge graphs and source code for the paper:

Brack A., Müller D., Hoppe A., Ewerth R. (2021) Coreference Resolution in Research Papers from Multiple Domains, ECIR 2021 (accepted for publication). 
- Submitted version: https://t.co/IlZhlcAgmv?amp=1

## Installation
Python 3.8 required.  Install the requirements with:
- pip install requirements.txt

## Datasets

- data/stm-coref: contains the annotated coreferences separated per domain BRAT/standoff format.
- data/stm-entities: contains the annotated concepts separated per domain in BRAT/standoff format (from https://github.com/arthurbra/stm-corpus)
- data/silver_labelled: contains predicted concepts and coreferences of the silver labelled corpus
- data/STM coreference annotation guidelines.pdf: annotation guidelines for coreference annotions in the STM-corpus

## Knowledge Graphs
The folder knowledge_graph/ contains various knowledge graphs.

### Test-STM-KG
- knowledge_graph/gold_kg_cross_domain.jsonl: Test-STM-KG (cross-domain)
- knowledge_graph/gold_kg_in_domain.jsonl: Test-STM-KG (in-domain)
- knowledge_graph/entity_resolution_annotations.tsv: links mentions to Wiktionary and Wikipedia (from https://gitlab.com/TIBHannover/orkg/orkg-nlp/-/tree/master/STEM-ECR-v1.0)

To evaluate the effect of coreference resolution in knowledge graph population against the Test-STM-KG, run the following python script:
- evaluate_kgs_against_test_stm_kg.py: prints the evaluation metrics

### In-domain and cross-domain research knowledge graph 
- knowledge_graph/stm_silver_kg_cross_domain_with_corefs.jsonl: Contains the cross-domain KG populated from 55,485 abstracts
- knowledge_graph/stm_silver_kg_in_domain_with_corefs.jsonl: Contains the in-domain KG populated from 55,485 abstracts
- knowledge_graph/top domain-specific concepts.xlsx: contains the most frequent domain-specific concepts per concept type and domain

### Build knowledge graphs
To build the in-domain and cross-domain Test-STM-KG and the research knowledge graphs, run the following python script:
- build_kgs.py: creates the knowledge graphs in knowledge_graph/



TODO: commit scripts to train the coreference model
