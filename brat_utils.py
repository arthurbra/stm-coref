# from __future__ import annotations # auskommentiert, damit der code in google colab läuft. Wenn local läuft, dann check bei compare to, das coprus2: Corpus
import os
from collections import defaultdict
import re
from typing import Tuple, List
import json

from utils import get_files_in_folder, flatten


def print_clusters(clusters, text, return_output=False):
    output = ''
    uses_word_indices = not isinstance(text, str)
    if uses_word_indices:
        text = flatten(text)

    for index, cluster in enumerate(clusters):
        output += f'c{index}: '
        for (start, end) in cluster:
            if uses_word_indices:
                output += f'{text[start:end + 1]} | '
            else:
                output += f'{text[start:end]} | '
        output += '\n'

    if return_output:
        return output
    else:
        print(output)


def brat_extract_properties(doc_path, allow_fragments=True):
    entities = {}  # T?(entity identifier) -> (type, (start, end))
    clusters = defaultdict(list)
    text = ''

    if os.path.exists(doc_path + '.txt'):
        with open(doc_path + '.txt', mode='r') as reader:
            text = reader.read()

    if os.path.exists(doc_path + '.ann'):
        with open(doc_path + '.ann', mode='r') as file:
            for line in file:
                if '\t' not in line:
                    continue

                annotation_type, annotation = line[0:line.index('\t')], line[line.index('\t') + 1:]
                is_entity_entry, is_relation_entry = annotation_type[0] == 'T', annotation_type[0] == 'R'

                if is_entity_entry:
                    col0, _ = annotation.split('\t')
                    entity_type = col0.split()[0]

                    all_indices = [int(x) for x in re.findall(r'\d+', col0[col0.index(' ') + 1:])]
                    uses_fragments = ';' in col0
                    if allow_fragments and uses_fragments:  # e.g. orginal STM-corpus: Chemistry/S1388248113001951
                        second_fragment_precedes_first_fragment = all_indices[2] < all_indices[1]
                        if second_fragment_precedes_first_fragment:
                            entity_indices = all_indices[2], all_indices[1]
                        else:
                            entity_indices = all_indices[0], all_indices[3]
                    else:
                        entity_indices = tuple(all_indices)

                    if 'Cluster' in entity_type:
                        cluster_index = re.search(r'\d+$', entity_type).group()
                        clusters[cluster_index].append(entity_indices)
                    else:
                        entities[annotation_type] = (entity_type, entity_indices)

    clusters = list(clusters.values())
    entities = list(sorted(entities.values(), key=lambda x: x[1][0]))

    return text, entities, clusters


class Document(object):
    DOMAIN_TO_DOMAIN_ID = {  # domain id is used in the genre embeddings
        'Agriculture': 'ag',
        'Astronomy': 'as',
        'Biology': 'bi',
        'Chemistry': 'ch',
        'Computer_Science': 'cs',
        'Earth_Science': 'es',
        'Engineering': 'en',
        'Materials_Science': 'ms',
        'Mathematics': 'ma',
        'Medicine': 'me'
    }

    def __init__(self, doc_path, allow_fragments, doc_with_entities_path=None, is_from_kg_corpus=False):
        self.path = doc_path
        self.is_annotated = doc_with_entities_path is not None
        if self.is_annotated:
            self.text, _, self.clusters = brat_extract_properties(doc_path, allow_fragments)
            _, self.entities, _ = brat_extract_properties(doc_with_entities_path, allow_fragments)
        else:
            self.text, _, _ = brat_extract_properties(doc_path)
            self.entities, self.clusters = [], []
        self.name = doc_path.split('/')[-1]
        if not is_from_kg_corpus:
            self.domain = doc_path.split('/')[-2]
            self.key = self.DOMAIN_TO_DOMAIN_ID[self.domain] + '_' + self.name
        else:
            self.domain = 'Materials_Science'  # since no domain is given for these abstracts, will use this as default
            self.key = 'ms' + '_' + self.name

    def print_clusters(self):
        print_clusters(self.clusters, self.text)

    def print_entities(self):
        e_type_to_e_phrases = defaultdict(list)
        for e_type, (start, end) in self.entities:
            e_type_to_e_phrases[e_type].append(self.text[start:end])
        for e_type, e_phrases in e_type_to_e_phrases.items():
            print(f'entity type: {e_type}')
            for e_phrase in e_phrases:
                print(f'{e_phrase} | ', end='')
            print()

    def __repr__(self):
        return f'{self.domain}/{self.key}'


class Corpus(object):
    def __init__(self, corpus_folder_path, is_kg_corpus=False):
        doc_paths = [doc_path.replace('.txt', '') for doc_path in get_files_in_folder(corpus_folder_path, pattern='*.txt')]
        self.docs = [Document(doc_path, allow_fragments=True, is_from_kg_corpus=is_kg_corpus) for doc_path in doc_paths]
        self._doc_name_to_doc = {doc.name: doc for doc in self.docs}
        self._doc_key_to_doc = {doc.key: doc for doc in self.docs}

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def __getitem__(self, item):
        if type(item) is str:
            if not self._doc_name_to_doc.get(item) is None:
                return self._doc_name_to_doc[item]
            else:
                return self._doc_key_to_doc[item]
        else:
            return self.docs[item]

    def __repr__(self):
        return f'number of documents: {len(self.docs)}\n'

    def __len__(self):
        return len(self.docs)


class STMCorpus(Corpus):
    def __init__(self, cluster_corpus_folder_path, entity_corpus_folder_path, allow_fragments=True,
                 filter_out_irrelevant_entity_types=False, irrelevant_entity_types=None):
        super().__init__(cluster_corpus_folder_path)

        irrelevant_entity_types_default = ['Result', 'Object', 'Task']
        if not irrelevant_entity_types:
            irrelevant_entity_types = irrelevant_entity_types_default

        doc_paths = zip(get_files_in_folder(cluster_corpus_folder_path, pattern='*.txt'),
                        get_files_in_folder(entity_corpus_folder_path, pattern='*.txt'))
        doc_paths = [(p1.replace('.txt', ''), p2.replace('.txt', '')) for p1, p2 in doc_paths]
        self.docs = [Document(c_doc_path, allow_fragments, e_doc_path) for c_doc_path, e_doc_path in doc_paths]
        if not allow_fragments:  # sciee can't handle mentions which occur in fragments
            for doc in self.docs:
                formatted_clusters = []
                for cluster in doc.clusters:
                    # filter out the clusters which now only contain one mention
                    if len(cluster) > 1:
                        formatted_clusters.append(cluster)
                doc.clusters = formatted_clusters

        if filter_out_irrelevant_entity_types:
            for doc in self.docs:
                e_indices_to_e_type = {e_indices: e_type for e_type, e_indices in doc.entities}
                doc.clusters = [
                    fc for fc in [
                        [m for m in c if e_indices_to_e_type.get(m) not in irrelevant_entity_types]
                        for c in doc.clusters] if len(fc) >= 2
                ]
                doc.entities = [(e_type, e_indices) for e_type, e_indices in doc.entities if e_type not in irrelevant_entity_types]

        self._doc_name_to_doc = {doc.name: doc for doc in self.docs}
        self._doc_key_to_doc = {doc.key: doc for doc in self.docs}

    def get_train_dev_test(self, fold: int = 0, folds_fp: str = 'data/stm_coref_folds.json') -> Tuple[List[Document], List[Document], List[Document]]:
        with open(folds_fp) as file:
            folds = json.load(file)

        def to_docs(instances: List[str]) -> List[Document]:
            return [self[doc_str.split('/')[1]] for doc_str in instances]

        return to_docs(folds[f'fold_{fold}']['train']), \
               to_docs(folds[f'fold_{fold}']['dev']), \
               to_docs(folds[f'fold_{fold}']['test'])