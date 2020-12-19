# STM-corpus with coreference resolution and knowledge graph population

This repository contains the annotated STM corpus with coreferences, populated knowledge graphs and source code for the paper:

Brack A., Müller D., Hoppe A., Ewerth R. (2021) Coreference Resolution in Research Papers from Multiple Domains, ECIR 2021 (accepted for publication). 
- Submitted version: https://t.co/IlZhlcAgmv?amp=1

## Installation
Python 3.8 required.  Install the requirements with:
- pip install requirements.txt

## Datasets

- data/stm-coref: contains the annotated coreferences separated per domain BRAT/standoff format.
- data/stm-entities: contains the annotated concepts separated per domain in BRAT/standoff format (see https://github.com/arthurbra/stm-corpus)
- data/silver_labelled: contains predicted concepts and coreferences of the silver labelled corpus

## Create Knowledge Graphs
To build the in-domain and cross-domain Test-STM-KG and the research knowledge graphs, run the following python script:
- build_kgs.py: creates the knowledge graphs in knowledge_graph/

## Evaluate Knowledge Graphs against Test-STM-KG
To evaluate the effect of coreference resolution against the Test-STM-KG, run the following python script:
- evaluate_kgs_against_test_stm_kg.py: prints the evaluation metrics


TODO: commit scripts to train the coreference model
