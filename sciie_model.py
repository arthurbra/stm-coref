import os
from threading import Thread
import shutil
import json

import nltk
from nltk import word_tokenize, sent_tokenize

from brat_utils import STMCorpus, Corpus
import utils
from index_converter import IndexConverter
from config import Config

nltk.download("punkt")


class SCIIEModel:
    def __init__(self, corpus: Corpus = Config.STM_CORPUS, fold: int = 0, use_pretrained_model: bool = False):
        self.corpus = corpus
        self.fold = fold
        self.use_pretrained_model = use_pretrained_model
        self.experiment = 'SCIIE_SciERC' if use_pretrained_model else 'SCIIE_STM'
        self.is_setup = False

    def _setup(self) -> None:
        os.chdir(Config.SCIIE_DIR)

        embeddings_dir = 'embeddings'
        logs_dir = 'logs'
        embeddings_dl_links = ['http://nlp.stanford.edu/data/glove.840B.300d.zip',
                             'https://dada.cs.washington.edu/qasrl/data/glove_50_300_2.zip']
        pretrained_model_dl_link = 'http://nlp.cs.washington.edu/sciIE/models/scientific_best_coref.zip'

        # clean up the logs-dir from previous experiments
        if os.path.exists(logs_dir):
            shutil.rmtree(logs_dir)
        os.mkdir(logs_dir)

        def get_dir(url: str, parent_dir: str) -> str:
            return os.path.join(parent_dir, os.path.basename(url).replace('.zip', ''))

        def download_file(url: str, dest: str) -> None:
            os.system(f'wget -P {dest} {url}')
            os.system(f'unzip {get_dir(url, dest)}.zip -d {dest}')
            os.remove(get_dir(url, dest) + '.zip')

        if not os.path.exists(embeddings_dir):
            os.mkdir(embeddings_dir)
            print('Downloading embeddings.')
            for dl_link in embeddings_dl_links:
                download_file(dl_link, embeddings_dir)
            print('Downloads finished.')

        if self.use_pretrained_model and not os.path.exists(get_dir(pretrained_model_dl_link, Config.SCIIE_DIR)):
            print('Downloading pretrained model.')
            download_file(pretrained_model_dl_link, Config.SCIIE_DIR)
            print('Download finished.')

        if self.use_pretrained_model:
            shutil.copytree(src=get_dir(pretrained_model_dl_link, Config.SCIIE_DIR), dst=logs_dir)

        # creates train-, dev-, test.json files
        if os.path.exists(os.path.join(Config.SCIIE_DIR, 'data')):
            shutil.rmtree(os.path.join(Config.SCIIE_DIR, 'data'))  # delete previous elmo embeddings and dataset-splits
        os.makedirs(os.path.join(Config.SCIIE_DIR, 'data/processed_data/json'))

        if isinstance(self.corpus, STMCorpus):
            folds_fp = os.path.join(Config.DATA_DIR, 'stm_coref_folds.json')
        else:
            folds_fp = os.path.join(Config.DATA_DIR, 'scierc_folds.json')
        self._create_json_files(self.corpus, self.fold, folds_fp,
                                output_dir=os.path.join(Config.SCIIE_DIR, 'data/processed_data/json'))

        # generate elmo embeddings for the given train-dev-test split which will be stored at data/processed_data/elmo
        print('Creating elmo-embeddings and storing them at data/processed_data/elmo.')
        os.makedirs(os.path.join(Config.SCIIE_DIR, 'data/processed_data/elmo'))
        utils.execute(['python3', 'scripts/filter_embeddings.py', 'embeddings/glove.840B.300d.txt',
                       'embeddings/glove.840B.300d.txt.filtered',
                       'data/processed_data/json/train.json', 'data/processed_data/json/dev.json'])
        utils.execute(['python3', 'scripts/get_char_vocab.py'])
        for split in ['train', 'dev', 'test']:
            utils.execute(['python3', 'generate_elmo.py',
                           '--input', os.path.join(Config.SCIIE_DIR, f'data/processed_data/json/{split}.json'),
                           '--output', os.path.join(Config.SCIIE_DIR, f'data/processed_data/elmo/{split}.hdf5')])

        # saves Prec-, Rec- and F1-scores for each domain on the test-set to EVAL_RESULTS_DIR
        os.environ['eval_results_fp'] = os.path.join(Config.EVAL_RESULTS_DIR, f'{self.experiment}_{self.fold}_eval.csv')

    def train(self) -> None:
        """
        Trains for a fixed number of epochs. Continuously evaluates on the dev-set while the training is running.
        """
        if not self.is_setup:
            self._setup()
        os.chdir(Config.SCIIE_DIR)

        evaluator_thread = Thread(target=lambda: utils.execute(['python3', 'evaluator.py', 'scientific_best_coref'])) # TODO besser als funktion / partial? warum wird nicht abgebrochen, warum nicht in EvalResults geschrieben?
        trainer_thread = Thread(target=lambda: utils.execute(['python3', 'singleton.py', 'scientific_best_coref']))

        print('Starting training.')
        trainer_thread.start()
        evaluator_thread.start()

        # blocks until the training has finished (after 300 Epochs)
        trainer_thread.join()

        # evaluator runs indefinitely until 'stop_sciie_evaluator' is set to True
        os.environ['stop_sciie_evaluator'] = 'True'
        evaluator_thread.join()

        print('Training finished.')

    def evaluate(self):
        """
        Evaluates on the test-set. Saves Precision-, Recall- and F1-Scores per domain to a csv-file at EVAL_RESULTS_DIR.
        """
        if not self.is_setup:
            self._setup()
        os.chdir(Config.SCIIE_DIR)

        if not os.path.exists(Config.EVAL_RESULTS_DIR):
            os.mkdir(Config.EVAL_RESULTS_DIR)

        utils.execute(['python3', 'test_single.py', 'test_scientific_best_coref'])

    def predict(self, input_json_fp: str, output_dir: str):
        raise NotImplementedError('Will be implemented in the future.')

        if not self.is_setup:
            self._setup()
        os.chdir(SCIIE_DIR)
        # TODO create embeddings
        changes = {
            'output_path': os.path.join(output_dir, os.path.basename(input_json_fp) + ''),
            'eval_path': input_json_fp,
            'lm_path_dev': input_json_fp.replace('.json', '.hdf5')
        }
        utils.change_conf_params('scientific_best_coref', f'{SCIIE_DIR}/experiments.conf', changes)
        utils.execute(['python3', 'write_single.py', 'scientific_best_coref'])

    @staticmethod
    def _create_json_files(corpus: Corpus, fold: int, folds_fp: str, output_dir: str) -> None:
        for data_set, docs in zip(['train', 'dev', 'test'], corpus.get_train_dev_test(folds_fp, fold)):
            with open(os.path.join(output_dir, data_set + '.json'), mode='w', encoding='utf-8') as writer:
                for doc in docs:
                    tokenized_text = [word_tokenize(s) for s in sent_tokenize(doc.text)]
                    empty_list = [["" for _ in sentence] for sentence in tokenized_text]

                    # convert char indices to word indices
                    converter = IndexConverter(doc.text, utils.flatten(tokenized_text))
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
                        # "ner": formatted_entities # somehow throws an exception
                        "ner": []
                    }
                    writer.write(json.dumps(batch) + '\n')

        print(f'created train, dev, test.json at {output_dir}.')
