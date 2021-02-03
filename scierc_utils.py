import os
from collections import defaultdict
import random
import json
import nltk
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from brat_utils import Corpus
from index_converter import IndexConverter
from utils import get_files_in_folder, flatten

nltk.download("punkt")


def prepare_corpus(corpus, fold, output_folder_path='../SciERC/data/processed_data/json'):
    for data_set, docs in zip(['train', 'dev', 'test'], corpus.get_train_dev_test(fold=fold)):
        with open(os.path.join(output_folder_path, data_set + '.json'), mode='w', encoding='utf-8') as writer:
            for doc in docs:
                tokenized_text = [word_tokenize(s) for s in sent_tokenize(doc.text)]
                empty_list = [["" for _ in sentence] for sentence in tokenized_text]

                # convert char indices to word indices
                converter = IndexConverter(doc.text, flatten(tokenized_text))
                converted_clusters = []
                for c in doc.clusters:
                    l = []
                    for m in c:
                        try:
                            l.append(converter.to_word_index(m))
                        except:
                            pass
                    if len(l) >= 2:
                        converted_clusters.append(l)

                sentence_to_start_end = {}  # inclusive, exclusive indices
                curr_sentence_index = 0
                sentence_start_index = 0
                for char_index, char in enumerate(doc.text):
                    if char == '\n':  # must be the sentence end
                        sentence_to_start_end[curr_sentence_index] = (sentence_start_index, char_index)
                        curr_sentence_index += 1
                        sentence_start_index = char_index + 1

                batch = {
                    "doc_key": doc.key,
                    "clusters": converted_clusters,
                    "sentences": tokenized_text,
                    "speakers": empty_list,
                    # "relations": []
                    # "ner": formatted_entities # schmeisst leider ne exception
                    "ner": []
                }
                writer.write(json.dumps(batch) + '\n')

    print(f'created train, dev, test.json at {output_folder_path}.')


def __brat_extract_properties(doc,
                              docs_use_coref_relations):
    entities = {}  # T?(entity identifier) -> (type, (start, end))
    clusters = [] if docs_use_coref_relations else defaultdict(list)

    with open(doc + '.txt', mode='r') as reader:
        text = reader.read()

    if os.path.exists(doc + '.ann'):
        with open(doc + '.ann', mode='r') as file:
            for line in file:
                if '\t' not in line:
                    print('no tab', doc, line)
                    continue
                annotation_type, annotation = line[0:line.index('\t')], line[line.index('\t') + 1:]
                is_entity_entry, is_relation_entry = annotation_type[0] == 'T', annotation_type[0] == 'R'

                if docs_use_coref_relations:
                    if is_entity_entry:
                        try:
                            col0, col1 = annotation.split('\t')
                            entity_type, entity_indices = col0.split()[0], (
                                int(col0.split()[1]), int(col0.split()[2]))
                            entities[annotation_type] = (entity_type, entity_indices)
                        except ValueError:
                            print(
                                f'ValueError, __brat_extract_properties, handle this doc: {doc}, line: {line}')  # TODO temp
                    elif is_relation_entry:
                        relation_type, arg1, arg2 = annotation.split()
                        arg1, arg2 = arg1.split(':')[1], arg2.split(':')[1]  # Arg1:T1 Arg2:T3 ->(T1, T3)
                        arg1, arg2 = entities[arg1][1], entities[arg2][1]  # to indices T1-> (14, 16) T3...

                        # TEMP
                        text1 = text[arg1[0]:arg1[1]]
                        text2 = text[arg2[0]:arg2[1]]

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

                else:
                    if is_entity_entry:
                        col0, col1 = annotation.split('\t')
                        entity_type = col0.split()[0]
                        if ';' in col0:  # e.g. orginal STM-cropus: Chemistry/S1388248113001951, when fragments are used in brat
                            entity_indices = int(col0.split()[1]), int(col0.split()[3])
                        else:
                            entity_indices = (int(col0.split()[1]), int(col0.split()[2]))

                        if 'Cluster' in entity_type:
                            cluster_index = re.search(r'\d+$', entity_type).group()
                            clusters[cluster_index].append(entity_indices)

                        else:
                            entities[annotation_type] = (entity_type, entity_indices)

    if docs_use_coref_relations:
        # the entity identities were only used for creating the clusters
        entities = list(entities.values())

        def merge(lists):  # mit 'S0167739X13001349' verstehen
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

    else:
        entities = list(entities.values())
        clusters = list(clusters.values())

    return text, entities, clusters


def __read_scierc_corpus(corpus_fp):
    doc_paths = [doc_path.replace('.txt', '') for doc_path in get_files_in_folder(corpus_fp, pattern='*.txt')]
    doc_key_to_text_entities_clusters = {}
    for fp in doc_paths:
        doc_key = os.path.basename(fp).split('.')[0]
        text, entities, clusters = __brat_extract_properties(fp, docs_use_coref_relations=True)
        doc_key_to_text_entities_clusters[doc_key] = (text, entities, clusters)

    return doc_key_to_text_entities_clusters


def __create_folds(doc_key_to_text_entities_clusters, k=10):
    # scierc contains 500 docs. The train-, dev- and test-set contain 350, 50, 100 docs respectively
    doc_keys = list(doc_key_to_text_entities_clusters.keys())
    random.shuffle(doc_keys)
    # split doc keys into k=10 groups of equal size, 50 docs per split
    k_groups = [doc_keys[round(i / k * len(doc_keys)):round((i / k + 0.1) * len(doc_keys))] for i in range(k)]

    fold_to_train_dev_test_docs = {}
    for fold in range(k):
        test = flatten(k_groups[fold: (fold + 2)] if fold != (k - 1) else [k_groups[(k - 1)], k_groups[0]])
        dev = k_groups[(fold + 2) % k]
        train = list(set(doc_keys).difference(set().union(dev, test)))
        fold_to_train_dev_test_docs[str(fold)] = (train, dev, test) # str(fold) because json converts dict-key integers to strings

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


def prepare_scierc_corpus(corpus_fp, fold, folds_fp='data/scierc_folds.json', output_fp='../BertForCorefRes/bert_data',
                          use_predefined_split=False, predefined_split_files_fp='data/scierc_split/processed_data/json',
                          num_docs_reduction_to_percent=100, reduced_fold_fp='data/scierc_to_{}_percent_reduced_fold.json'):
    """ Creates train, dev and test .conll (conll 2011/2012 format) files for the given fold.
    Downlaod scierc_corpus from http://nlp.cs.washington.edu/sciIE/data/sciERC_raw.tar.gz.
    num_docs_reduction_to_percent = 20: reduces the num of docs in the split to 20% of its original size. """
    doc_key_to_text_entities_clusters = __read_scierc_corpus(corpus_fp)
    if use_predefined_split:
        print('Reading predefined split.')
        train_dev_test_docs = []
        for file_name in ['train.json', 'dev.json', 'test.json']:
            with open(os.path.join(predefined_split_files_fp, file_name), 'r', encoding='UTF-8') as file:
                doc_keys = [json.loads(line)['doc_key'] for line in file.readlines()]
                train_dev_test_docs.append(doc_keys)
        train, dev, test = train_dev_test_docs
    else:
        if not os.path.exists(folds_fp):
            print(f'Creating 10 new folds.')
            fold_to_train_dev_test_docs = __create_folds(doc_key_to_text_entities_clusters)
            with open(folds_fp, 'w') as file:
                json.dump(fold_to_train_dev_test_docs, file)
        else:
            print(f'Reading fold: {fold} from file: {folds_fp}.')
            with open(folds_fp, 'r') as file:
                fold_to_train_dev_test_docs = json.load(file)
        train, dev, test = fold_to_train_dev_test_docs[str(fold)]

    if num_docs_reduction_to_percent < 100:
        if os.path.exists(reduced_fold_fp.format(num_docs_reduction_to_percent)):
            print('Reading reduced split from file.')
            # so all experiments / folds can use the exact same split
            with open(reduced_fold_fp.format(num_docs_reduction_to_percent), 'r') as file:
                train, dev, test = json.load(file)
        else:
            print(f'Creating new reduced split and storing it at {reduced_fold_fp.format(num_docs_reduction_to_percent)}.')
            random.shuffle(train); random.shuffle(dev); random.shuffle(test)
            train = train[:round(len(train) * num_docs_reduction_to_percent / 100)]
            dev = dev[:round(len(dev) * num_docs_reduction_to_percent / 100)]
            test = test[:round(len(test) * num_docs_reduction_to_percent / 100)]
            with open(reduced_fold_fp.format(num_docs_reduction_to_percent), 'w') as file:
                json.dump((train, dev, test), file)

    if not os.path.exists(output_fp):
        os.mkdir(output_fp)
    sep = 4 * ' '  # separator
    output_paths = [os.path.join(output_fp, name + '.english.v4_gold_conll') for name in ['train', 'dev', 'test']]
    for doc_keys, output_path in zip([train, dev, test], output_paths):
        with open(output_path, mode='w') as output_file:
            for doc_key in doc_keys:
                text, entities, clusters = doc_key_to_text_entities_clusters[doc_key]
                tokenized_text = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
                # tokenized_text = word_tokenize(doc.text) # müsste eigentlich so gehen, aber .conll sind dann nur buchstabe pro zeile
                converter = IndexConverter(text, flatten(tokenized_text))
                wi_clusters = []

                for c in clusters:
                    l = []
                    for m in c:
                        try:
                            l.append(converter.to_word_index(m))
                        except:
                            # in very few cases nltk doesn't tokenize the way IndexConverter expects it to
                            # thus IndexConverter throws an exception
                            pass
                    if len(l) >= 2:  # checks if the cluster still contains at least two mentions
                        wi_clusters.append(l)

                wi_to_cluster_str = {}
                for cluster_index, cluster in enumerate(wi_clusters):
                    for wi_start, wi_end in cluster:
                        if wi_start == wi_end:
                            if wi_start not in wi_to_cluster_str:
                                wi_to_cluster_str[wi_start] = f'({cluster_index})'
                            else:
                                wi_to_cluster_str[wi_start] += f'|({cluster_index})'
                        else:
                            if wi_start not in wi_to_cluster_str:
                                wi_to_cluster_str[wi_start] = f'({cluster_index}'
                            else:
                                wi_to_cluster_str[wi_start] += f'|({cluster_index}'

                            if wi_end not in wi_to_cluster_str:
                                wi_to_cluster_str[wi_end] = f'{cluster_index})'
                            else:
                                wi_to_cluster_str[wi_end] += f'|{cluster_index})'
                output_file.write(f'#begin document ({doc_key}); part 000\n')

                wi = 0
                for sentence in tokenized_text:
                    for index, word in enumerate(sentence):
                        cluster_str = wi_to_cluster_str[wi] if wi in wi_to_cluster_str else '-'

                        output_file.write(doc_key + sep
                                          + '0' + sep
                                          + str(index) + sep
                                          + word + sep
                                          + 8 * ('-' + sep)
                                          + cluster_str + '\n')

                        wi += 1
                    output_file.write('\n')
                output_file.write('#end document\n')

    print(f'Created .conll files from the SciERC-Corpus of size: train={len(train)}, dev={len(dev)}, test={len(test)}.')
