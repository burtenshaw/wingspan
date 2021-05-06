#%%
import os
import torch
import logging
import torch
import time
import pandas as pd
import numpy as np

from collections import OrderedDict
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from maxfw.model import MAXModelWrapper
os.chdir('/home/burtenshaw/code/dialogue_dash/toxic')
from core.model import ModelWrapper
from core.bert_pytorch import BertForMultiLabelSequenceClassification, InputExample, convert_examples_to_features
from config import DEFAULT_MODEL_PATH, LABEL_LIST, MODEL_META_DATA as model_meta
#%%
logger = logging.getLogger()
BASE_DIR = '/home/burtenshaw/now/spans_toxic'


class IBMToxicity(ModelWrapper):

    MODEL_META_DATA = model_meta

    def __init__(self, path=DEFAULT_MODEL_PATH):
        """Instantiate the BERT model."""
        logger.info('Loading model from: {}...'.format(path))

        # Load the model
        # 1. set the appropriate parameters
        self.eval_batch_size = 32
        self.max_seq_length = 256
        self.do_lower_case = True

        # 2. Initialize the PyTorch model
        model_state_dict = torch.load(DEFAULT_MODEL_PATH + 'pytorch_model.bin', map_location='cpu')
        self.tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_PATH, do_lower_case=self.do_lower_case)
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH,
                                                                                num_labels=len(LABEL_LIST),
                                                                                state_dict=model_state_dict)
        self.device = torch.device('cpu')
        self.model.to(self.device)

        # 3. Set the layers to evaluation mode
        self.model.eval()

        # logger.info('Loaded model')

ibm_model_wrapper = IBMToxicity()
#%%

for fold_n in ['3', '4']:

    pred_dir = os.path.join(BASE_DIR, 'predictions', fold_n)
    data_dir = os.path.join(BASE_DIR, 'data', fold_n)
    
    df = pd.read_pickle(os.path.join(data_dir, 'test.bin'))
    tox = ibm_model_wrapper.predict(df.text.to_list())
    pd.DataFrame(tox, index = df.index).to_pickle(os.path.join(pred_dir, 'ibm_tox.bin'))


# %%
