import os
from collections import defaultdict
import re
from typing import Tuple, List, Dict
import json
import random

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

    def __init__(self, name: str, text: str, clusters: List[List[Tuple[int, int]]], domain: str = None,
                 entities: List[Tuple[str, Tuple[int, int]]] = None):
        self.name = name
        self.text = text
        self.domain = domain if domain else 'Materials_Science'  # will use this as default
        self.key = self.DOMAIN_TO_DOMAIN_ID[self.domain] + '_' + self.name
        self.entities = entities
        self.clusters = clusters

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
    def __init__(self, docs: List[Document]):
        self.docs = docs
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

    def get_train_dev_test(self, folds_fp: str, fold: int = 0) \
            -> Tuple[List[Document], List[Document], List[Document]]:
        raise NotImplementedError('Must be defined when inheriting from Corpus.')

    @staticmethod
    def _extract_properties(corpus_dir: str) \
            -> Tuple[str, List[Tuple[str, Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        raise NotImplementedError('Must be defined when inheriting from Corpus.')


class STMCorpus(Corpus):
    def __init__(self, cluster_corpus_folder_path: str, entity_corpus_folder_path: str, allow_fragments: bool = True,
                 filter_out_irrelevant_entity_types: bool = False, irrelevant_entity_types: List[str] = None):
        irrelevant_entity_types_default = ['Result', 'Object', 'Task']
        if not irrelevant_entity_types:
            irrelevant_entity_types = irrelevant_entity_types_default

        docs = []
        for doc_entities_fp, doc_clusters_fp in zip(get_files_in_folder(cluster_corpus_folder_path, pattern='*.txt'),
                                                    get_files_in_folder(entity_corpus_folder_path, pattern='*.txt')):
            text, _, clusters = self._extract_properties(doc_clusters_fp, allow_fragments)
            _, entities, _ = self._extract_properties(doc_entities_fp, allow_fragments)
            domain, name = doc_clusters_fp.replace('.txt', '').split('/')[-2:]
            docs.append(Document(name, text, clusters, domain, entities))

        if not allow_fragments:  # sciee can't handle mentions which occur in fragments
            for doc in docs:
                formatted_clusters = []
                for cluster in doc.clusters:
                    # filter out the clusters which now only contain one mention
                    if len(cluster) > 1:
                        formatted_clusters.append(cluster)
                doc.clusters = formatted_clusters

        if filter_out_irrelevant_entity_types:
            for doc in docs:
                e_indices_to_e_type = {e_indices: e_type for e_type, e_indices in doc.entities}
                doc.clusters = [
                    fc for fc in [
                        [m for m in c if e_indices_to_e_type.get(m) not in irrelevant_entity_types]
                        for c in doc.clusters] if len(fc) >= 2
                ]
                doc.entities = [(e_type, e_indices) for e_type, e_indices in doc.entities if
                                e_type not in irrelevant_entity_types]

        super().__init__(docs)

    def get_train_dev_test(self, folds_fp: str, fold: int = 0) -> Tuple[List[Document], List[Document], List[Document]]:
        with open(folds_fp) as file:
            folds = json.load(file)

        def to_docs(instances: List[str]) -> List[Document]:
            return [self[doc_str.split('/')[1]] for doc_str in instances]

        return to_docs(folds[f'fold_{fold}']['train']), \
               to_docs(folds[f'fold_{fold}']['dev']), \
               to_docs(folds[f'fold_{fold}']['test'])

    @staticmethod
    def _extract_properties(corpus_dir: str, allow_fragments: bool = True) -> \
            Tuple[str, List[Tuple[str, Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        entities = {}  # T?(entity identifier) -> (type, (start, end))
        clusters = defaultdict(list)
        text = ''

        if os.path.exists(corpus_dir + '.txt'):
            with open(corpus_dir + '.txt', mode='r') as reader:
                text = reader.read()

        if os.path.exists(corpus_dir + '.ann'):
            with open(corpus_dir + '.ann', mode='r') as file:
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


class SciercCorpus(Corpus):
    def __init__(self, corpus_fp: str, use_predefined_split: bool = False,
                 predefined_split_files_fp: str = '../data/scierc_split/processed_data/json',
                 num_docs_reduction_to_percent: int = 100,
                 reduced_fold_fp: str = '../data/scierc_to_{}_percent_reduced_fold.json'):
        """
        Download the brat documents needed for this class from http://nlp.cs.washington.edu/sciIE/data/sciERC_raw.tar.gz.
        Download the json-files containing the original train-, dev-, test-split from
        http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz if you want to set use_predefined_split to True.

        :param corpus_fp: directory containing the brat files from the scierc-corpus.
        :param use_predefined_split: if True: get_train_dev_test() will use the original train-, dev-, test-split
        contained in the second link above, if False: will create 10 different folds and use them in get_train_dev_test().
        :param predefined_split_files_fp: directory where the json-files with the original split are stored.
        :param num_docs_reduction_to_percent: num_docs_reduction_to_percent = 20: reduces the num of docs in the split
        to 20% of its original size.
        :param reduced_fold_fp: if num_docs_reduction_to_percent smaller than 100: will reduce the number of docs in the
         split and store the results at this file-path.
        """
        self.corpus_fp = corpus_fp
        self.use_predefined_split = use_predefined_split
        self.predefined_split_files_fp = predefined_split_files_fp
        self.num_docs_reduction_to_percent = num_docs_reduction_to_percent
        self.reduced_fold_fp = reduced_fold_fp

        doc_paths = [doc_path.replace('.txt', '') for doc_path in get_files_in_folder(corpus_fp, pattern='*.txt')]
        docs = []
        for fp in doc_paths:
            doc_key = os.path.basename(fp).split('.')[0]
            text, entities, clusters = self._extract_properties(fp)
            doc = Document(name=doc_key, text=text, clusters=clusters, entities=entities)
            doc.key = doc_key
            docs.append(doc)

        super().__init__(docs)

    def get_train_dev_test(self, folds_fp: str, fold: int = 0) -> Tuple[List[Document], List[Document], List[Document]]:
        if self.use_predefined_split:
            print('Reading predefined split.')
            train_dev_test_docs = []
            for file_name in ['train.json', 'dev.json', 'test.json']:
                with open(os.path.join(self.predefined_split_files_fp, file_name), 'r', encoding='UTF-8') as file:
                    doc_keys = [json.loads(line)['doc_key'] for line in file.readlines()]
                    train_dev_test_docs.append(doc_keys)
            train, dev, test = train_dev_test_docs
        else:
            if not os.path.exists(folds_fp):
                print(f'Creating 10 new folds and storing them at: {folds_fp}.')
                fold_to_train_dev_test_docs = self._create_folds(self.docs)
                with open(folds_fp, 'w') as file:
                    json.dump(fold_to_train_dev_test_docs, file)
            else:
                print(f'Reading fold: {fold} from file: {folds_fp}.')
                with open(folds_fp, 'r') as file:
                    fold_to_train_dev_test_docs = json.load(file)
            train, dev, test = fold_to_train_dev_test_docs[str(fold)]

        if self.num_docs_reduction_to_percent < 100:
            if os.path.exists(self.reduced_fold_fp.format(self.num_docs_reduction_to_percent)):
                print('Reading reduced split from file.')
                # so all experiments / folds can use the exact same split
                with open(self.reduced_fold_fp.format(self.num_docs_reduction_to_percent), 'r') as file:
                    train, dev, test = json.load(file)
            else:
                print(f'Creating new reduced split and storing it '
                      f'at {self.reduced_fold_fp.format(self.num_docs_reduction_to_percent)}.')
                random.shuffle(train), random.shuffle(dev), random.shuffle(test)
                train = train[:round(len(train) * self.num_docs_reduction_to_percent / 100)]
                dev = dev[:round(len(dev) * self.num_docs_reduction_to_percent / 100)]
                test = test[:round(len(test) * self.num_docs_reduction_to_percent / 100)]
                with open(self.reduced_fold_fp.format(self.num_docs_reduction_to_percent), 'w') as file:
                    json.dump((train, dev, test), file)

        train_docs = [doc for doc in self.docs if doc.key in train]
        dev_docs = [doc for doc in self.docs if doc.key in dev]
        test_docs = [doc for doc in self.docs if doc.key in test]

        return train_docs, dev_docs, test_docs

    @staticmethod
    def _create_folds(docs: List[Document], num_folds: int = 10) -> \
            Dict[str, Tuple[List[str], List[str], List[str]]]:
        # scierc contains 500 docs. The train-, dev- and test-set contain 350, 50, 100 docs respectively
        doc_keys = [doc.key for doc in docs]
        random.shuffle(doc_keys)
        # split fp keys into num_folds=10 groups of equal size, 50 docs per split
        n_groups = [doc_keys[round(i / num_folds * len(doc_keys)):
                             round((i / num_folds + 0.1) * len(doc_keys))] for i in range(num_folds)]

        fold_to_train_dev_test_docs = {}
        for fold in range(num_folds):
            test = flatten(n_groups[fold:(fold + 2)] if fold != (num_folds - 1) else [n_groups[(num_folds - 1)], n_groups[0]])
            dev = n_groups[(fold + 2) % num_folds]
            train = list(set(doc_keys).difference(set().union(dev, test)))
            # str(fold) because json converts dict-key integers to strings
            fold_to_train_dev_test_docs[str(fold)] = (train, dev, test)

        # check that each test-set of a the folds is unique
        assert (not any(t1 == t2 for dk1, (_, _, t1) in fold_to_train_dev_test_docs.items() for dk2, (_, _, t2) in
                        fold_to_train_dev_test_docs.items() if dk1 != dk2))
        # check for each fold that every doc_key is either part of the train-, dev- or the test-set
        assert (not any(len(set().union(train, dev, test)) != len(doc_keys) for train, dev, test in
                        fold_to_train_dev_test_docs.values()))
        # check that no doc_key is repeated in the same fold, e.g. is part of test-set AND train-set
        assert (not any(len(train) + len(dev) + len(test) != len(doc_keys) for train, dev, test in
                        fold_to_train_dev_test_docs.values()))

        return fold_to_train_dev_test_docs

    @staticmethod
    def _extract_properties(corpus_dir: str) \
            -> Tuple[str, List[Tuple[str, Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        entities = {}  # T?(entity identifier) -> (type, (start, end))
        clusters = []

        with open(corpus_dir + '.txt', mode='r') as reader:
            text = reader.read()

        if os.path.exists(corpus_dir + '.ann'):
            with open(corpus_dir + '.ann', mode='r') as file:
                for line in file:
                    if '\t' not in line:
                        print('no tab', corpus_dir, line)
                        continue
                    annotation_type, annotation = line[0:line.index('\t')], line[line.index('\t') + 1:]
                    is_entity_entry, is_relation_entry = annotation_type[0] == 'T', annotation_type[0] == 'R'

                    if is_entity_entry:
                        try:
                            col0, col1 = annotation.split('\t')
                            entity_type, entity_indices = col0.split()[0], (
                                int(col0.split()[1]), int(col0.split()[2]))
                            entities[annotation_type] = (entity_type, entity_indices)
                        except ValueError:
                            print(
                                f'ValueError, __brat_extract_properties, handle this fp: {corpus_dir}, line: {line}')  # TODO temp
                    elif is_relation_entry:
                        relation_type, arg1, arg2 = annotation.split()
                        arg1, arg2 = arg1.split(':')[1], arg2.split(':')[1]  # Arg1:T1 Arg2:T3 ->(T1, T3)
                        arg1, arg2 = entities[arg1][1], entities[arg2][1]  # to indices T1-> (14, 16) T3...

                        if relation_type == 'COREF':
                            was_found_in_cluster = False

                            for cluster in clusters:
                                if arg1 in cluster and arg2 not in cluster:
                                    cluster.append(arg2)
                                    was_found_in_cluster = True
                                elif arg2 in cluster and arg1 not in cluster:
                                    cluster.append(arg1)
                                    was_found_in_cluster = True
                                elif arg1 in cluster and arg2 in cluster:
                                    was_found_in_cluster = True

                            if not was_found_in_cluster:
                                clusters.append([arg1, arg2])
                        else:
                            pass  # other relations are not implemented

        # the entity identities were only used for creating the clusters
        entities = list(entities.values())

        def merge(lists):
            item_to_index = defaultdict(list)
            for i, l in enumerate(lists):
                for item in l:
                    item_to_index[item].append(i)

            merges = []
            for item, index in item_to_index.items():
                if len(index) > 1 and index not in merges:
                    merges.append(index)

            if len(merges) == 0:
                return lists
            else:
                merged_lists = []
                merges = merge(merges)
                for indices_to_merge in merges:
                    merged_list = []
                    for index in indices_to_merge:
                        for item in lists[index]:
                            if item not in merged_list:
                                merged_list.append(item)
                    merged_lists.append(merged_list)

                for index, l in enumerate(lists):
                    if index not in flatten(merges):
                        merged_lists.append(l)

                return merged_lists

        clusters = merge(clusters)

        return text, entities, clusters
