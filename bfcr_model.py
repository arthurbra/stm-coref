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
from brat_utils import Corpus, KGCorpus, Document, STMCorpus
from index_converter import IndexConverter
from predictions_reader import read_predictions, Model

nltk.download("punkt")

BFCR_FP = os.path.abspath('BertForCorefRes')
BERT_DATA_FP = os.path.abspath(os.path.join(BFCR_FP, 'bert_data'))
EVAL_RESULTS_FP = os.path.abspath('EvalResults')

Checkpoint = namedtuple('Checkpoint', 'folder dl_link')
ExperimentConfig = namedtuple('ExperimentConfig', 'name vocab_folder ckpts')


class CKPTS(Enum):
    SCIBERT = Checkpoint(folder='scibert_scivocab_uncased', dl_link='https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz')
    SPANBERT = Checkpoint(folder='spanbert_hf_base', dl_link='https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz')
    SPANBERT_ONTO = Checkpoint(folder='spanbert_base', dl_link='http://nlp.cs.washington.edu/pair2vec/spanbert_base.tar.gz')
    SPANBERT_ONTO_STM = Checkpoint(folder='spanbert_base_stm', dl_link='https://drive.google.com/uc?id=1wx5aIjRKP9BwyQB1bYAuGTxeySmbeQsm')
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
                                                               vocab_folder=CKPTS.SPANBERT_ONTO_STM.folder,
                                                               ckpts=[CKPTS.SPANBERT_ONTO_STM])
}


def execute(command: List[str]) -> None:
    # prefix = '!' if BFCRModel.RUN_IN_IPYTHON else ''
    # os.system(prefix + command[0] + ' '.join(command[1:]))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if not BFCRModel.ONLY_SHOW_STDERR:
        for line in process.stdout:
            print(line, end='')
        for line in process.stderr:
            print(line, end='')
    else:
        for line in process.stderr:
            print(line, end='')

    # while process.poll() is None:
    #     out = process.stdout.readline()
    #     if out != '':
    #         print(out, end='')
    #     err = process.stderr.readline()
    #     if err != '':
    #         print(err, end='')

# print(line, end='')
    # for line in iter(lambda: process.stdout.readline() + '\n' + process.stderr.readline(), ''):
    #     print(line, end='')
    # for line in process.stderr:
    #     print(line, end='')


class BFCRModel:
    STM_COREF_CORPUS_FP = os.path.abspath('data/stm-coref')
    STM_ENTITIES_CORPUS_FP = os.path.abspath('data/stm-entities')
    STM_CORPUS = STMCorpus(STM_COREF_CORPUS_FP, STM_ENTITIES_CORPUS_FP)
    ONLY_SHOW_STDERR = False

    def __init__(self, experiment: Experiment = Experiment.BFCR_Span_Onto_STM_pretrained,
                 corpus: STMCorpus = STM_CORPUS, fold: int = 0, seed: int = 0, max_seg_len: int = 384):
        self.experiment = experiment
        self.experiment_config = EXPERIMENT_TO_CONFIG[experiment]
        self.corpus = corpus
        self.fold = fold
        self.seed = seed
        self.max_seg_len = max_seg_len
        self.is_setup = False

    def _setup(self, checkpoints_folder: str = os.path.join(BFCR_FP, 'checkpoints')):
        # cleanup the bert_data-folder
        os.system(f'rm -r {BERT_DATA_FP}')

        # download checkpoints which are used for the first time
        for ckpt in self.experiment_config.ckpts:
            if not os.path.exists(os.path.join(checkpoints_folder, ckpt.folder)):
                print(f'Downloading file "{ckpt.folder}" to "{checkpoints_folder}".')
                if ckpt.dl_link.startswith('https://drive.google.com/'):
                    gdown.download(ckpt.dl_link, output=os.path.join(checkpoints_folder, ckpt.folder), quiet=False)
                else:
                    os.system(f'wget -P {checkpoints_folder} {ckpt.dl_link}')
                ending = '.tar.gz' if ckpt.dl_link.endswith('.tar.gz') else '.zip'
                if ending == '.zip':
                    os.system(f'unzip {os.path.join(checkpoints_folder, ckpt.folder)}.zip')
                else:
                    os.system(f'tar xvzf {os.path.join(checkpoints_folder, ckpt.folder)}.tar.gz -C {checkpoints_folder}')
                os.system(f'rm {os.path.join(checkpoints_folder, ckpt.folder)}{ending}')
                print('Download complete.')

            # move the checkpoints necessary for the experiment to the checkpoints-folder
            os.mkdir(BERT_DATA_FP)
            os.system(f'cp -av {os.path.join(checkpoints_folder, ckpt.folder)} {os.path.join(BERT_DATA_FP, ckpt.folder)}')

        os.chdir(BFCR_FP)  # BFCR_FP contains the python-scripts, which are used in train(), evaluate(), predict()

        # creates train, test, dev.conll
        folds_fp = '../data/stm_coref_folds.json'
        for partition, file_name in zip(self.corpus.get_train_dev_test(self.fold, folds_fp), ['train', 'dev', 'test']):
            texts = [doc.text for doc in partition]
            doc_keys = [doc.key for doc in partition]
            clusters = [doc.clusters for doc in partition]
            self._create_conll_file(texts, file_name, doc_keys, output_folder=BERT_DATA_FP, clusters=clusters)

        # creates train, dev, test.jsonlines,
        # where the texts are split into segments with a maximum length of max_seg_len
        vocab_fp = os.path.abspath(os.path.join(BERT_DATA_FP, self.experiment_config.vocab_folder, 'vocab.txt'))
        input_dir = output_dir = BERT_DATA_FP
        execute(['python3', 'minimize.py', vocab_fp, input_dir, output_dir, 'False', str(self.max_seg_len)])

        utils.set_seed_value(self.seed)
        os.environ['eval_results_fp'] = os.path.join(
            EVAL_RESULTS_FP,
            f'{self.experiment.name}_{self.fold}_s{self.seed}_msl_{self.max_seg_len}'
        )
        os.environ['data_dir'] = BERT_DATA_FP
        self.is_setup = True

    def train(self) -> None:
        if not self.is_setup:
            self._setup()

        if self.experiment not in [Experiment.BFCR_Span_Onto, Experiment.BFCR_Span_Onto_scierc_eval_only]:
            # train on train-set and find the best checkpoint by evaluating on dev-set
            changes = {
                'train_path': '${data_dir}/train.english.' + str(self.max_seg_len) + '.jsonlines',
                'eval_path': '${data_dir}/dev.english.' + str(self.max_seg_len) + '.jsonlines',
                'conll_eval_path': '${data_dir}/dev.english.v4_gold_conll',
                'max_segment_len': str(self.max_seg_len)
            }
            utils.change_conf_params(self.experiment_config.name, f'{BFCR_FP}/experiments.conf', changes)
            execute(['python', 'train.py', self.experiment_config.name])
        else:
            print(f'Experiment: {self.experiment} is not meant to be trained! Use another Experiment if you want '
                  f'to train.')

    def evaluate(self) -> None:
        if not self.is_setup:
            self._setup()

        if not os.path.exists(EVAL_RESULTS_FP):
            os.mkdir(EVAL_RESULTS_FP)

        # evaluate on test-set
        changes = {
            'train_path': '${data_dir}/train.english.' + str(self.max_seg_len) + '.jsonlines',
            'eval_path': '${data_dir}/test.english.' + str(self.max_seg_len) + '.jsonlines',
            'conll_eval_path': '${data_dir}/test.english.v4_gold_conll',
            'max_segment_len': str(self.max_seg_len)
        }
        utils.change_conf_params(self.experiment_config.name, f'{BFCR_FP}/experiments.conf', changes)
        execute(['python', 'evaluate.py', self.experiment_config.name])

    def predict(self, texts: List[str] = None, domains: List[str] = None, kg_corpus: KGCorpus = None,
                predictions_fp: str = os.path.join(BERT_DATA_FP, 'predictions.jsonlines'),
                remove_predictions_file: bool = True, create_standoff_annotations: bool = False,
                standoff_annotations_dir: str = '../data/coref_predictions_standoff'):
        if not self.is_setup:
            self._setup()

        if (not texts and not domains and not kg_corpus) or (texts and kg_corpus):
            raise Exception('Must define either texts or kg_corpus (but not both)!')

        if not domains:
            domains = ['Computer_Science' for _ in texts]

        if any(domain not in Document.DOMAIN_TO_DOMAIN_ID for domain in domains):
            raise Exception(f'Each domain must be one of these: {Document.DOMAIN_TO_DOMAIN_ID.values()}')

        if kg_corpus:
            texts = [d.text for d in kg_corpus]
            doc_keys = [d.key for d in kg_corpus]
        else:
            doc_keys = [Document.DOMAIN_TO_DOMAIN_ID[domain] + '_' + str(i) for i, domain in enumerate(domains)]

        input_file_name = 'texts_to_predict'
        self._create_conll_file(texts, input_file_name, doc_keys, output_folder=BERT_DATA_FP)

        # creates a .jsonlines-file, where the texts are split into segments with a maximum length of max_seg_len
        vocab_fp = os.path.abspath(os.path.join(BERT_DATA_FP, self.experiment_config.vocab_folder, 'vocab.txt'))
        input_dir = output_dir = BERT_DATA_FP
        execute(['python3', 'minimize.py', vocab_fp, input_dir, output_dir, 'False', str(self.max_seg_len), input_file_name])  # TODO oder nur python?

        # make sure the correct segment length is contained in the experiments.config
        changes = {'max_segment_len': str(self.max_seg_len)}
        utils.change_conf_params(self.experiment_config.name, f'{BFCR_FP}/experiments.conf', changes)

        input_fp = os.path.join(BERT_DATA_FP, f'{input_file_name}.english.{self.max_seg_len}.jsonlines')
        execute(['python3', 'predict.py', self.experiment_config.name, input_fp, predictions_fp])

        all_predicted_clusters = read_predictions(predictions_fp, texts, doc_keys, used_model=Model.BFCR)

        if remove_predictions_file:
            os.remove(input_fp)

        if create_standoff_annotations:
            if os.path.exists(standoff_annotations_dir):
                shutil.rmtree(standoff_annotations_dir)
            os.mkdir(standoff_annotations_dir)

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
                           clusters: List[List[Tuple[int, int]]] = None) -> None:
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
