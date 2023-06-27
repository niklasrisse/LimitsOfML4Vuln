import os
import sys
import random
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import List
import argparse

from clang import *
from clang import cindex

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from transformers import PLBartForSequenceClassification, AutoTokenizer

from transformations import *
from models import MarkovModel

TRANSFORMATIONS = [no_transformation, tf_1, tf_2, tf_3, tf_4, tf_5, tf_6, tf_7, tf_8, tf_9, tf_12, tf_11, tf_10, tf_13]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print(device)

# deterministic settings for exact reproducibility
seedlist = [42, 834, 692, 489, 901, 408, 819, 808, 531, 166]
fixed_seed = 489 #random.choice(seedlist)

os.environ['PYTHONHASHSEED'] = str(fixed_seed)
torch.manual_seed(fixed_seed)
torch.cuda.manual_seed(fixed_seed)
torch.cuda.manual_seed_all(fixed_seed)
np.random.seed(fixed_seed)
random.seed(fixed_seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

machine = "localhost" if os.path.dirname(os.path.realpath(__file__)).split("/")[1] == "Users" else "cluster"

if machine == "cluster":
    pathprefix = "icse2024_experiments/"
else:
    pathprefix = ""


train_index=set()
valid_index=set()
test_index=set()

with open('./' + pathprefix + 'datasets/CodeXGLUE/train-IDs.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))

with open('./' + pathprefix + 'datasets/CodeXGLUE/valid-IDs.txt') as f:
    for line in f:
        line=line.strip()
        valid_index.add(int(line))

with open('./' + pathprefix + 'datasets/CodeXGLUE/test-IDs.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))


mydata = pd.read_json('./' + pathprefix + 'datasets/CodeXGLUE/CodeXGLUE.json')
developer_commit_data_test = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Test.json')

codexglue_train=mydata.iloc[list(train_index)]
codexglue_valid=mydata.iloc[list(valid_index)]
codexglue_test=mydata.iloc[list(test_index)]
vpp_test = developer_commit_data_test

my_tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-c-cpp-defect-detection")

encodings_train = my_tokenizer.batch_encode_plus(codexglue_train.func, add_special_tokens=False)
    
mm = MarkovModel(encodings_train["input_ids"], np.array(list(my_tokenizer.get_vocab().values()) + [-1, -2]))
mm.inject_noise(0.00001)

def apply_transformation(df, all_trafos, trafo_to_apply, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply_in_random_selection = None):
    
    if trafo_to_apply.__name__ == "tf_11":
        df.func = df.func.apply(trafo_to_apply, args=(all_trafos, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply_in_random_selection))
    elif trafo_to_apply.__name__ == "tf_10":
        df.func = df.func.apply(trafo_to_apply, args=(training_set_sample_neg,))
    elif trafo_to_apply.__name__ == "tf_13":
        df.func = df.func.apply(trafo_to_apply, args=(training_set_sample_pos,))
    else:
        df.func = df.func.apply(trafo_to_apply)
        
    return df

def save_list_to_file(lst, filename):
    with open(filename, 'wb') as file:
        pickle.dump(lst, file)
        
m1_not_vulnerable = codexglue_train[codexglue_train["target"] == 0]
training_set_sample_neg = m1_not_vulnerable.iloc[0]["func"]

m1_vulnerable = codexglue_train[codexglue_train["target"] == 1]
training_set_sample_pos = m1_vulnerable.iloc[0]["func"]
        
for tf_func in TRANSFORMATIONS:
    
    sample_ces_test = []
    
    codexglue_test=mydata.iloc[list(test_index)]

    codexglue_test_transformed = apply_transformation(codexglue_test, TRANSFORMATIONS, tf_func, training_set_sample_neg, training_set_sample_pos, tf_func)
    encodings_transformed = my_tokenizer.batch_encode_plus(codexglue_test_transformed.func, add_special_tokens=False)
    progress_bar = tqdm(range(len(encodings_transformed["input_ids"])))

    for sample in encodings_transformed["input_ids"]:
        sample_ce = mm.calculate_cross_entropy_of_sample(sample)
        sample_ces_test.append(sample_ce)
        
        progress_bar.update(1)

    save_list_to_file(sample_ces_test,'./' + pathprefix + 'results/naturalness_codexglue_test_{}.pkl'.format(tf_func.__name__))

x=234