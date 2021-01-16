import json
from enum import Enum
from typing import List, Tuple

from utils import flatten
from brat_utils import Document
from index_converter import IndexConverter
from nltk import sent_tokenize, word_tokenize


class Model(Enum):
    BFCR, SCIIE = range(2)


def read_predictions(predictions_fp: str, texts: List[str], doc_keys: List[str], used_model: Model) -> List[List[List[Tuple[int, int]]]]:
    # note: uses doc_keys to access texts, tokenized_texts and predicted_clusters instead of just enumerating them
    # in case the entries of the predictions.jsonlines aren't in the same order as in the parameters
    # as then some texts would get assigned the wrong coref-predictions when doc_keys aren't used
    doc_key_to_text = {doc_key: text for doc_key, text in zip(doc_keys, texts)}

    with open(predictions_fp, mode='r') as file:
        entries = [json.loads(line) for line in file]

        if used_model == Model.BFCR:
            if any(e['doc_key'][-2:] != '_0' for e in entries):
                raise NotImplementedError()

            doc_key_to_predicted_clusters = {e['doc_key'][:-2]: e['predicted_clusters'] for e in entries}
            doc_key_to_tokenized_text = {e['doc_key'][:-2]: flatten(e['sentences']) for e in entries}

            # removes the '[CLS]'  '[SEP]' from the start, end respectively, converts tokes such as '##up' to 'up'
            formatted_doc_key_to_tokenized_text = {}
            for doc_key, tokenized_text in doc_key_to_tokenized_text.items():
                formatted_doc_key_to_tokenized_text[doc_key] = list(
                    map(lambda s: s.replace('##', ''), tokenized_text[1:-1]))

            # since '[CLS]' now is missing, all token indices are wrong by one, -> need to shift them to the left by one
            formatted_doc_key_to_pred_clusters = {}
            for doc_key, clusters in doc_key_to_predicted_clusters.items():
                formatted_doc_key_to_pred_clusters[doc_key] = []
                for c in clusters:
                    l = []
                    for start, end in c:
                        l.append((start - 1, end - 1))
                    formatted_doc_key_to_pred_clusters[doc_key].append(l)

            doc_key_to_tokenized_text = formatted_doc_key_to_tokenized_text
            doc_key_to_pred_clusters = formatted_doc_key_to_pred_clusters

        elif used_model == Model.SCIIE:
            doc_key_to_pred_clusters = {e['doc_key']: e['coref'] for e in entries}

            # the tokenized text ist not included in the predictions.jsonlines-file
            doc_key_to_tokenized_text = {doc_key: [word_tokenize(s) for s in sent_tokenize(text)]
                                         for doc_key, text in doc_key_to_text.items()}

        # convert the word indices to character indices
        for doc_key, text in doc_key_to_text.items():
            tokenized_text = doc_key_to_tokenized_text[doc_key]
            converter = IndexConverter(text, flatten(tokenized_text))
            predicted_clusters = []
            for cluster in doc_key_to_pred_clusters[doc_key]:
                # quite a lot of the letters aren't found in the scivocab. Thus when the bert tokenizer tries to
                # tokenize words such as "Mg3NF3" it returns a "[UNK]" Token, since 'M' isn't part of the scivocab
                # it's impossible to find the original word before it became an "[UNK]"-Token, so these "[UNK]"-Tokens
                # need to be filtered out, when part of a cluster
                filtered_cluster = []
                for m in cluster:
                    if '[UNK]' not in tokenized_text[m[0]:m[1] + 1]:
                        filtered_cluster.append(m)
                if len(filtered_cluster) < 2:  # as a result of the filtering
                    continue

                try:
                    predicted_clusters.append(converter.to_char_index(filtered_cluster))
                except:
                    continue
            doc_key_to_pred_clusters[doc_key] = predicted_clusters

        return [doc_key_to_pred_clusters[doc_key] for doc_key in doc_keys]

