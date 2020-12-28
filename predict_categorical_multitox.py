#%%
import os
import torch
import logging
import torch
import time
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, roc_curve
from collections import OrderedDict
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from maxfw.model import MAXModelWrapper
os.chdir('/home/burtenshaw/code/dialogue_dash/toxic')
from core.model import ModelWrapper
from core.bert_pytorch import BertForMultiLabelSequenceClassification, InputExample, convert_examples_to_features
from config import DEFAULT_MODEL_PATH, LABEL_LIST, MODEL_META_DATA as model_meta

logger = logging.getLogger()

#%%
data_dir = '/home/burtenshaw/now/spans_toxic/data/'
output_dir = data_dir
tox_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

df = pd.read_pickle(data_dir + 'train.bin')

#%%

class DataPredict(ModelWrapper):

    MODEL_META_DATA = model_meta

    def __init__(self, path=DEFAULT_MODEL_PATH):
        """Instantiate the BERT model."""
        logger.info('Loading model from: {}...'.format(path))

        # Load the model
        # 1. set the appropriate parameters
        self.eval_batch_size = 64
        self.max_seq_length = 256
        self.do_lower_case = True

        # 2. Initialize the PyTorch model
        model_state_dict = torch.load(DEFAULT_MODEL_PATH+'pytorch_model.bin', map_location='cpu')
        self.tokenizer = BertTokenizer.from_pretrained(DEFAULT_MODEL_PATH, do_lower_case=self.do_lower_case)
        self.model = BertForMultiLabelSequenceClassification.from_pretrained(DEFAULT_MODEL_PATH,
                                                                             num_labels=len(LABEL_LIST),
                                                                             state_dict=model_state_dict)
        self.device = torch.device('cuda')
        self.model.to(self.device)

        # 3. Set the layers to evaluation mode
        self.model.eval()

        # logger.info('Loaded model')
model_wrapper = DataPredict()
# %%
result = model_wrapper.predict(df.text.to_list())
#%%
_result = pd.DataFrame(result)
df = df.merge(_result.add_suffix('_ibm_pred'), left_index=True, right_index=True)
#%%
df.to_pickle(output_dir + 'TRAIN_IBM_TOX.csv')


# %%

