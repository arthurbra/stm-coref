import os
import shutil
from collections import namedtuple
from enum import Enum
import subprocess
from typing import List, Tuple
import shutil

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gdown

import utils
from brat_utils import Corpus, Document, STMCorpus
from index_converter import IndexConverter
from predictions_reader import read_predictions, Model
from config import Config

nltk.download("punkt")

Checkpoint = namedtuple('Checkpoint', 'folder dl_link')
ExperimentConfig = namedtuple('ExperimentConfig', 'name vocab_folder ckpts')


class CKPTS(Enum):
    SCIBERT = Checkpoint(folder='scibert_scivocab_uncased', dl_link='https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz')
    SPANBERT = Checkpoint(folder='spanbert_hf_base', dl_link='https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz')
    SPANBERT_ONTO = Checkpoint(folder='spanbert_base', dl_link='http://nlp.cs.washington.edu/pair2vec/spanbert_base.tar.gz')
    SPANBERT_ONTO_STM = Checkpoint(folder='BFCR_Span_Onto_STM', dl_link='https://drive.google.com/uc?id=1wx5aIjRKP9BwyQB1bYAuGTxeySmbeQsm')
    BERT = Checkpoint(folder='cased_L-12_H-768_A-12', dl_link='https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip')

    @property
    def folder(self):
        return self.value.folder

    @property
    def dl_link(self):
        return self.value.dl_link


class Experiment(Enum):
    BFCR_Span_Onto = 1
    BFCR_Span_Onto_STM = 2
    BFCR_Span_STM = 3
    BFCR_Sci_STM = 4
    BFCR_Span_Onto_scierc_eval_only = 5
    BFCR_Span_Onto_STM_pretrained = 6


EXPERIMENT_TO_CONFIG = {
    Experiment.BFCR_Span_Onto: ExperimentConfig(name='spanbert_base',
                                                vocab_folder=CKPTS.SPANBERT_ONTO.folder,
                                                ckpts=[CKPTS.SPANBERT_ONTO]),
    Experiment.BFCR_Span_Onto_STM: ExperimentConfig(name='spanbert_base',
                                                    vocab_folder=CKPTS.SPANBERT_ONTO.folder,
                                                    ckpts=[CKPTS.SPANBERT_ONTO]),
    Experiment.BFCR_Span_STM: ExperimentConfig(name='train_spanbert_base',
                                               vocab_folder=CKPTS.SPANBERT_ONTO.folder,
                                               ckpts=[CKPTS.BERT, CKPTS.SPANBERT, CKPTS.SPANBERT_ONTO]), # CKPTS.SPANBERT_ONTO only for the vocab-file
    Experiment.BFCR_Sci_STM: ExperimentConfig(name='train_scibert_base',
                                              vocab_folder=CKPTS.SCIBERT.folder,
                                              ckpts=[CKPTS.SCIBERT]),
    Experiment.BFCR_Span_Onto_STM_pretrained: ExperimentConfig(name='spanbert_base',
                                                               vocab_folder=CKPTS.SPANBERT_ONTO.folder,
                                                               ckpts=[CKPTS.SPANBERT_ONTO_STM])
}


class BFCRModel:
    DATA_DIR = os.path.abspath('data')
    STM_COREF_CORPUS_FP = os.path.abspath('data/stm-coref')
    STM_ENTITIES_CORPUS_FP = os.path.abspath('data/stm-entities')
    STM_CORPUS = STMCorpus(STM_COREF_CORPUS_FP, STM_ENTITIES_CORPUS_FP)

    def __init__(self, experiment: Experiment = Experiment.BFCR_Span_Onto_STM_pretrained,
                 corpus: Corpus = STM_CORPUS, fold: int = 0, seed: int = 0, max_seg_len: int = 384):
        self.experiment = experiment
        self.experiment_config = EXPERIMENT_TO_CONFIG[experiment]
        self.corpus = corpus
        self.fold = fold
        self.seed = seed
        self.max_seg_len = max_seg_len
        self.is_setup = False

    def _setup(self, checkpoints_folder: str = os.path.join(Config.BFCR_DIR, 'checkpoints')):
        # cleanup the bert_data-folder from previous experiments
        if os.path.exists(Config.BERT_DATA_DIR):
            os.system(f'rm -r {Config.BERT_DATA_DIR}')

        # download checkpoints which are used for the first time
        for ckpt in self.experiment_config.ckpts:
            if not os.path.exists(os.path.join(checkpoints_folder, ckpt.folder)):
                print(f'Downloading file "{ckpt.folder}" to "{checkpoints_folder}".')
                if not os.path.exists(checkpoints_folder):
                    os.mkdir(checkpoints_folder)

                if ckpt.dl_link.startswith('https://drive.google.com/'):
                    gdown.download(ckpt.dl_link, output=os.path.join(checkpoints_folder, ckpt.folder + '.zip'), quiet=False)
                else:
                    os.system(f'wget -P {checkpoints_folder} {ckpt.dl_link}')
                ending = '.tar.gz' if ckpt.dl_link.endswith('.tar.gz') else '.zip'
                if ending == '.zip':
                    os.system(f'unzip {os.path.join(checkpoints_folder, ckpt.folder)}.zip -d {checkpoints_folder}')
                else:
                    os.system(f'tar xvzf {os.path.join(checkpoints_folder, ckpt.folder)}.tar.gz -C {checkpoints_folder}')
                os.system(f'rm {os.path.join(checkpoints_folder, ckpt.folder)}{ending}')
                print('Download complete.')

            # move the checkpoints necessary for the experiment to the checkpoints-folder
            os.mkdir(Config.BERT_DATA_DIR)

            if self.experiment == Experiment.BFCR_Span_Onto_STM_pretrained:
                # BFCR_Span_Onto_STM_pretrained has the following name in the checkpoints-folder: spanbert_base_stm to
                # distinguish it from the other folders, but needs to have the name 'spanbert_base' in bert_data
                # since the experiments.conf refers to that folder
                dest = os.path.join(Config.BERT_DATA_DIR, 'spanbert_base')
            else:
                dest = os.path.join(Config.BERT_DATA_DIR, ckpt.folder)
            os.system(f'cp -av {os.path.join(checkpoints_folder, ckpt.folder)} {dest}')

        os.chdir(Config.BFCR_DIR)  # BFCR_FP is the path to the python-scripts, which are used in train(), evaluate(), predict()

        # creates train, test, dev.conll
        folds_fp = '../data/stm_coref_folds.json'
        for partition, file_name in zip(self.corpus.get_train_dev_test(folds_fp, self.fold), ['train', 'dev', 'test']):
            texts = [doc.text for doc in partition]
            doc_keys = [doc.key for doc in partition]
            clusters = [doc.clusters for doc in partition]
            self._create_conll_file(texts, file_name, doc_keys, output_folder=Config.BERT_DATA_DIR, clusters=clusters)

        # creates train, dev, test.jsonlines,
        # where the texts are split into segments with a maximum length of max_seg_len
        vocab_fp = os.path.abspath(os.path.join(Config.BERT_DATA_DIR, self.experiment_config.vocab_folder, 'vocab.txt'))
        input_dir = output_dir = Config.BERT_DATA_DIR
        utils.execute(['python3', 'minimize.py', vocab_fp, input_dir, output_dir, 'False', str(self.max_seg_len)])

        utils.set_seed_value(self.seed)
        os.environ['eval_results_fp'] = os.path.join(
            Config.EVAL_RESULTS_DIR,
            f'{self.experiment.name}_{self.fold}_s{self.seed}_msl_{self.max_seg_len}_eval.csv'
        )
        os.environ['data_dir'] = Config.BERT_DATA_DIR
        self.is_setup = True

    def train(self) -> None:
        if not self.is_setup:
            self._setup()

        os.chdir(Config.BFCR_DIR)  # BFCR_FP is the path to the python-scripts, which are used in train(), evaluate(), predict()

        if self.experiment not in [Experiment.BFCR_Span_Onto, Experiment.BFCR_Span_Onto_scierc_eval_only]:
            # train on train-set and find the best checkpoint by evaluating on dev-set
            changes = {
                'train_path': '${data_dir}/train.english.' + str(self.max_seg_len) + '.jsonlines',
                'eval_path': '${data_dir}/dev.english.' + str(self.max_seg_len) + '.jsonlines',
                'conll_eval_path': '${data_dir}/dev.english.v4_gold_conll',
                'max_segment_len': str(self.max_seg_len)
            }
            utils.change_conf_params(self.experiment_config.name, f'{Config.BFCR_DIR}/experiments.conf', changes)
            utils.execute(['python', 'train.py', self.experiment_config.name], show_stderr_first=True)
        else:
            print(f'Experiment: {self.experiment} is not meant to be trained! Use another Experiment if you want '
                  f'to train.')

    def evaluate(self) -> None:
        if not self.is_setup:
            self._setup()

        os.chdir(Config.BFCR_DIR)  # BFCR_FP is the path to the python-scripts, which are used in train(), evaluate(), predict()

        if not os.path.exists(Config.EVAL_RESULTS_DIR):
            os.mkdir(Config.EVAL_RESULTS_DIR)

        # evaluate on test-set
        changes = {
            'train_path': '${data_dir}/train.english.' + str(self.max_seg_len) + '.jsonlines',
            'eval_path': '${data_dir}/test.english.' + str(self.max_seg_len) + '.jsonlines',
            'conll_eval_path': '${data_dir}/test.english.v4_gold_conll',
            'max_segment_len': str(self.max_seg_len)
        }
        utils.change_conf_params(self.experiment_config.name, f'{Config.BFCR_DIR}/experiments.conf', changes)
        utils.execute(['python', 'evaluate.py', self.experiment_config.name])

    def predict(self, texts: List[str], domains: List[str] = None,
                predictions_fp: str = os.path.join(Config.BERT_DATA_DIR, 'predictions.jsonlines'),
                remove_input_file: bool = True, create_standoff_annotations: bool = False,
                standoff_annotations_dir: str = os.path.join(DATA_DIR, 'coref_predictions_standoff')):
        if not self.is_setup:
            self._setup()
        os.chdir(Config.BFCR_DIR)  # BFCR_FP is the path to the python-scripts, which are used in train(), evaluate(), predict()

        if not domains:
            domains = ['Computer_Science' for _ in texts]

        if any(domain not in Document.DOMAIN_TO_DOMAIN_ID for domain in domains):
            raise Exception(f'Each domain must be one of these: {Document.DOMAIN_TO_DOMAIN_ID.values()}')

        doc_keys = [Document.DOMAIN_TO_DOMAIN_ID[domain] + '_' + str(i) for i, domain in enumerate(domains)]

        input_file_name = 'texts_to_predict'
        self._create_conll_file(texts, input_file_name, doc_keys, output_folder=Config.BERT_DATA_DIR)

        # creates a .jsonlines-file, where the texts are split into segments with a maximum length of max_seg_len
        vocab_fp = os.path.abspath(os.path.join(Config.BERT_DATA_DIR, self.experiment_config.vocab_folder, 'vocab.txt'))
        input_dir = output_dir = Config.BERT_DATA_DIR
        utils.execute(['python3', 'minimize.py', vocab_fp, input_dir, output_dir, 'False', str(self.max_seg_len), input_file_name])

        # make sure the correct segment length is contained in the experiments.config
        changes = {'max_segment_len': str(self.max_seg_len)}
        utils.change_conf_params(self.experiment_config.name, f'{Config.BFCR_DIR}/experiments.conf', changes)

        input_fp = os.path.join(Config.BERT_DATA_DIR, f'{input_file_name}.english.{self.max_seg_len}.jsonlines')
        utils.execute(['python3', 'predict.py', self.experiment_config.name, input_fp, predictions_fp])

        all_predicted_clusters = read_predictions(predictions_fp, texts, doc_keys, used_model=Model.BFCR)

        if remove_input_file:
            os.remove(input_fp)

        if create_standoff_annotations:
            if os.path.exists(standoff_annotations_dir):
                shutil.rmtree(standoff_annotations_dir)
            os.makedirs(standoff_annotations_dir)

            shutil.copyfile(src=os.path.join(self.STM_COREF_CORPUS_FP, 'annotation.conf'),
                            dst=os.path.join(standoff_annotations_dir, 'annotation.conf'))
            shutil.copyfile(src=os.path.join(self.STM_COREF_CORPUS_FP, 'visual.conf'),
                            dst=os.path.join(standoff_annotations_dir, 'visual.conf'))

            for doc_key, text, predicted_clusters in zip(doc_keys, texts, all_predicted_clusters):
                file_name = os.path.join(standoff_annotations_dir, doc_key)

                with open(file_name + '.ann', mode='w') as file:
                    t_index = 1
                    start_end_to_t_index = {}
                    for c_i, cluster in enumerate(predicted_clusters):
                        for m_start, m_end in cluster:
                            file.write(f'T{t_index}\tCluster{c_i + 1} {m_start} {m_end}\t{text[m_start:m_end]}\n')
                            start_end_to_t_index[(m_start, m_end)] = t_index
                            t_index += 1

                with open(file_name + '.txt', mode='w') as file:
                    file.write(text)

        return all_predicted_clusters

    @staticmethod
    def _create_conll_file(texts: List[str], file_name: str, doc_keys: List[str], output_folder: str,
                           clusters: List[List[List[Tuple[int, int]]]] = None) -> None:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        sep = 4 * ' '  # separator
        output_fp = os.path.join(output_folder, f'{file_name}.english.v4_gold_conll')

        with open(output_fp, 'w') as file:
            for i, text in enumerate(texts):
                tokenized_text = [word_tokenize(sentence) for sentence in sent_tokenize(text)]

                if clusters:
                    converter = IndexConverter(text, utils.flatten(tokenized_text))
                    wi_clusters = []

                    for c in clusters[i]:
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

                tokenized_text = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
                file.write(f'#begin document ({doc_keys[i]}); part 000\n')

                wi = 0
                for sentence in tokenized_text:
                    for index, word in enumerate(sentence):
                        cluster_str = wi_to_cluster_str[wi] if clusters and wi in wi_to_cluster_str else '-'
                        file.write(doc_keys[i] + sep
                                   + '0' + sep
                                   + str(index) + sep
                                   + word + sep
                                   + 8 * ('-' + sep)
                                   + cluster_str + '\n')
                        wi += 1
                    file.write('\n')
                file.write('#end document\n')

        print(f'Created a .conll-file containing {len(texts)} docs at {output_fp}.')
