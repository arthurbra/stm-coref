[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coreference-resolution-in-research-papers/coreference-resolution-on-stm-coref)](https://paperswithcode.com/sota/coreference-resolution-on-stm-coref?p=coreference-resolution-in-research-papers)


# STM-corpus with coreference resolution and knowledge graph population

This repository contains the annotated STM corpus with coreferences, populated knowledge graphs and source code for the paper:

Brack A., Müller D., Hoppe A., Ewerth R. (2021) Coreference Resolution in Research Papers from Multiple Domains, ECIR 2021 (accepted for publication). 
- Preprint: https://arxiv.org/abs/2101.00884

## Installation
Python 3.7 required.  Install the requirements with:
- (in case pipenv isn't installed yet: `pip install pipenv`)
- `pipenv install`

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

## Coref-Models
The following two models were tested in the paper: [BFCR](https://github.com/mandarjoshi90/coref) 
and [SCIIE](https://github.com/YerevaNN/SciERC). Experiments with the two models on STM-coref, SciERC or any corpus 
that inherits from brat_utils.Corpus can be performed in the following ways:

#### BFCR
```python
model = BFCRModel()
```
The fold to train and evaluate on and the exact experiment involving BFCR can be specified with the
`fold` and `experiments` parameters. By default an already pretrained BFCR_Span_Onto_STM-model is used.

#### SCIIE
```python
model = SCIIEModel()
```

#### In general

BFCRModel and SCIIEModel automatically download the necessary checkpoints to run and store them locally for future use. 
By default BFCRModel and SCIIEModel use STM-coref, but [SciERC](http://nlp.cs.washington.edu/sciIE/) can also
be used. If you want to use SciERC, download it as explained in brat_utils.SciercCorpus.


### Training
Runs for a fixed number of epochs.
```python
model.train()
```
### Evaluation
```python
model.evaluate()
```
Evaluates on the test-set and prints the average MUC-, B³-, CEAFE- and CoNLL-scores. Also stores the average scores 
for the entire test-set and for each domain in the dataset as a .csv-file at stm-coref/EvalResults.

### Making Predictions
Coreference-clusters in texts can be predicted in the following two ways:
##### 1. ```Model.predict()```
```python
predicted_clusters = model.predict(texts, domains)  # domains optional
```
##### 2. using the ```predict.py```
Predict.py downloads the best performing model (BFCR_Span_Onto_STM), which is already pretrained and uses it to predict 
coreference-clusters of the texts stored in the input-file. The predictions are then stored in another file. Usage:

```pipenv run python predict.py "path/to/texts.jsonlines" "Optional-path/to/domains.jsonlines"```

## Visualize predictions:
The predictions can be visualized using [brat](https://brat.nlplab.org/). Creating the files needed for brat to visualize the predicted-clusters can be done in the two following ways:
```python
model.predict(texts, domains, create_standoff_annotations=True)
```
or \
```pipenv run python predict.py "path/to/texts.jsonlines" "Optional-path/to/domains.jsonlines" --create_standoff_annotations```
