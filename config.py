import os

from brat_utils import STMCorpus


class Config:
    DATA_DIR = os.path.abspath('data')
    EVAL_RESULTS_DIR = os.path.abspath('EvalResults')

    BFCR_DIR = os.path.abspath('BertForCorefRes')
    STM_COREF_CORPUS_DIR = os.path.abspath(os.path.join(DATA_DIR, 'stm-coref'))
    STM_ENTITIES_CORPUS_DIR = os.path.abspath(os.path.join(DATA_DIR, 'stm-entities'))
    BERT_DATA_DIR = os.path.abspath(os.path.join(BFCR_DIR, 'bert_data'))
    _STM_CORPUS = STMCorpus(STM_COREF_CORPUS_DIR, STM_ENTITIES_CORPUS_DIR, allow_fragments=False)

    SCIIE_DIR = os.path.abspath('SciERC')


