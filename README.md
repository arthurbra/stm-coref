# [Under Construction]

## Issues:
* requirements need to be fixed, pip install -r requirements.txt throws errors

## Workaround:
* runs in a Jupyter-Notebook in Google Colab

## Training

## Evaluation

## Make Predictions
Coreference-clusters in texts can be predicted in the following two ways:
##### 1. ```Model.predict()```
```python
model = BFCRModel()
predicted_clusters = model.predict(texts, domains)
```
##### 2. using the ```predict.py```
Predict.py downloads the best performing model (BFCR_Span_Onto_STM), which is already pretrained, uses it to predict coreference-clusters of the texts stored in the file given. 
```python predict.py "path/to/texts.jsonlines" "Optional-path/to/domains.jsonlines"```

## visualize predictions:
The predictions can be visualized using [brat](https://brat.nlplab.org/). Creating the files needed for brat to visualize the predicted-clusters can be done in the two following ways:
```python
model.predict(texts, domains, create_standoff_annotations=True)
```
or \
```python predict.py "path/to/texts.jsonlines" "Optional-path/to/domains.jsonlines" --create_standoff_annotations```