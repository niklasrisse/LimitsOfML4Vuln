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

from transformers import RobertaForSequenceClassification
from transformers import get_scheduler

from tokenizers import NormalizedString,PreTokenizedString
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers import processors
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import StripAccents, Replace
from tokenizers.processors import TemplateProcessing
from tokenizers.models import BPE

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from transformations import *

STR_TO_TRAFO = {
    "no_transformation" : no_transformation,
    "tf_1" : tf_1,
    "tf_2" : tf_2,
    "tf_3" : tf_3,
    "tf_4" : tf_4,
    "tf_5" : tf_5,
    "tf_6" : tf_6,
    "tf_7" : tf_7,
    "tf_8" : tf_8,
    "tf_9" : tf_9,
    "tf_10" : tf_10,
    "tf_11" : tf_11,
    "tf_12" : tf_12,
    "tf_13" : tf_13,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--random", default=False, required=False)
args = parser.parse_args()
random_guessing = args.random == "True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print(device)

# deterministic settings for exact reproducibility
seedlist = [42, 834, 692, 489, 901, 408, 819, 808, 531, 166]
fixed_seed = 408 #random.choice(seedlist)

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

class MyTokenizer:
    
    cidx = cindex.Index.create()
        

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        ## Tokkenize using clang
        tok = []
        tu = self.cidx.parse('tmp.c',
                       args=[''],  
                       unsaved_files=[('tmp.c', str(normalized_string.original))],  
                       options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()
            
            if spelling == '':
                continue
                
            ## Keyword no need

            ## Punctuations no need

            ## Literal all to BPE
            
            #spelling = spelling.replace(' ', '')
            tok.append(NormalizedString(spelling))

        return(tok)
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)
        
## Custom tokenizer

## Load pre-trained tokenizers
vocab, merges = BPE.read_file(vocab="./" + pathprefix + "models/tokenizer/drapgh-vocab.json", merges="./" + pathprefix + "models/tokenizer/drapgh-merges.txt")
my_tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

my_tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
my_tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
my_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
my_tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[
    ("<s>",0),
    ("<pad>",1),
    ("</s>",2),
    ("<unk>",3),
    ("<mask>",4)
    ]
)

my_tokenizer.enable_truncation(max_length=1024)
my_tokenizer.enable_padding(direction='right', pad_id=1, pad_type_id=0, pad_token='<pad>', length=None, pad_to_multiple_of=None)

def process_encodings(encodings):
    input_ids=[]
    attention_mask=[]
    for enc in encodings:
        input_ids.append(enc.ids)
        attention_mask.append(enc.attention_mask)
    return {'input_ids':input_ids, 'attention_mask':attention_mask}


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

m1=mydata.iloc[list(train_index)]
m2=mydata.iloc[list(valid_index)]
m3=mydata.iloc[list(test_index)]
m4 = developer_commit_data_test


def encodeDataframe(df):
    
    encodings = my_tokenizer.encode_batch(df.func)
    encodings = process_encodings(encodings)
    
    return encodings, df.target.tolist()

class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) ==  len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

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

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):

    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

def placeholder_injection(code):
    
    begin_of_function = code.index('{')

    code = code[0:begin_of_function + 1] + "\nchar placeholder\n" + code[begin_of_function + 1:]
    
    return code

    
TF_TO_APPLY_ON_TRAIN = placeholder_injection
PLACEHOLDER_TOKEN_ID = 49437

semantic_preserving_transformations = [no_transformation, tf_1, tf_2, tf_3, tf_4, tf_5, tf_6, tf_7, tf_8, tf_9, tf_12, tf_11, tf_10, tf_13]

m1_not_vulnerable = m1[m1["target"] == 0]
training_set_sample_neg = m1_not_vulnerable.iloc[0]["func"]

m1_vulnerable = m1[m1["target"] == 1]
training_set_sample_pos = m1_vulnerable.iloc[0]["func"]
    
m1 = apply_transformation(m1, semantic_preserving_transformations, TF_TO_APPLY_ON_TRAIN, training_set_sample_neg, training_set_sample_pos, None)

m4_changed = m4.drop('func', axis=1)
m4_changed = m4_changed.rename(columns={'changed_func': 'func'})
m4_changed = m4_changed[m4_changed["target"] == 1]
m4_changed = m4_changed.assign(target=0)
m4_changed = m4_changed.drop('new_path', axis=1)
m4_changed = m4_changed.drop('old_path', axis=1)
m4_changed = m4_changed.drop('new_commit', axis=1)
m4_changed = m4_changed.drop('old_commit', axis=1)

m4 = m4.drop('changed_func', axis=1)
m4 = m4.drop('new_path', axis=1)
m4 = m4.drop('old_path', axis=1)
m4 = m4.drop('new_commit', axis=1)
m4 = m4.drop('old_commit', axis=1)
m4 = m4[m4["target"] == 1]

print(m4_changed.count())
print(m4.count())

print(m4_changed.head())
print(m4.head())

train_encodings, train_labels = encodeDataframe(m1)
test_encodings, test_labels = encodeDataframe(m3)
fixed_encodings, fixed_labels = encodeDataframe(m4_changed)
not_fixed_encodings, not_fixed_labels = encodeDataframe(m4)

train_dataset = MyCustomDataset(train_encodings, train_labels)
test_dataset = MyCustomDataset(test_encodings, test_labels)
fixed_dataset = MyCustomDataset(fixed_encodings, fixed_labels)
not_fixed_dataset = MyCustomDataset(not_fixed_encodings, not_fixed_labels)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4)
fixed_dataloader = DataLoader(fixed_dataset, shuffle=False, batch_size=4)
not_fixed_dataloader = DataLoader(not_fixed_dataset, shuffle=False, batch_size=4)

print(len(train_dataloader))
print(len(test_dataloader))
print(len(fixed_dataloader))
print(len(not_fixed_dataloader))

num_epochs = 8
num_training_steps = num_epochs * len(train_dataloader)
num_testing_steps = num_epochs * len(test_dataloader)
num_fixed_steps = num_epochs * len(fixed_dataloader)
num_not_fixed_steps = num_epochs * len(not_fixed_dataloader)

model = RobertaForSequenceClassification.from_pretrained('./' + pathprefix + 'models/VulBERTa/')
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-6)
lr_scheduler = get_scheduler(
     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

cw = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=[0,1],y=m1.target.tolist())
    
c_weights = torch.FloatTensor([cw[0], cw[1]])

#criterion = torch.nn.CrossEntropyLoss() 
criterion = torch.nn.CrossEntropyLoss(weight=c_weights) 
criterion.to(device)

results = dict()

progress_bar = tqdm(range(num_training_steps + num_testing_steps * len(semantic_preserving_transformations) + num_fixed_steps + num_not_fixed_steps))

for epoch in range(num_epochs):
    
    model.train()
    
    losses = []
    predictions = []
    labels = []
    
    for batch in train_dataloader:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        original_input_ids = batch['input_ids']
            
        input_ids_as_list = batch["input_ids"].tolist()
        placeholder_positions = []
        for input_ids in input_ids_as_list:
            if input_ids.count(PLACEHOLDER_TOKEN_ID) == 1:
                placeholder_position = input_ids.index(PLACEHOLDER_TOKEN_ID)
                assert 0 <= placeholder_position < 1024
                placeholder_positions.append(placeholder_position)
                
        if len(placeholder_positions) == 4:
            
            optimizer.zero_grad()
            
            embedding_weights = model.roberta.embeddings.word_embeddings.weight
            input_ids_onehot = torch.nn.functional.one_hot(batch['input_ids'], num_classes=50000).to(torch.float)
            input_ids_onehot.requires_grad = True
            
            input_embeds = input_ids_onehot @ embedding_weights
            batch['position_ids'] = create_position_ids_from_input_ids(batch['input_ids'], 1)
            
            del(batch['input_ids'])
            
            batch['inputs_embeds'] = input_embeds
            
            outputs = model(**batch)
            
            logits = outputs.logits
            
            predicted_classes_original = torch.argmax(logits, dim=-1).tolist()
            
            loss = criterion(logits, batch["labels"])
        
            loss.backward()
            
            input_ids_onehot.requires_grad = False
            
            top_candidates = []
            
            for u, placeholder_position in enumerate(placeholder_positions):
            
                placeholder_one_hot = input_ids_onehot[u, placeholder_position, :]
                grad = input_ids_onehot.grad[u, placeholder_position, :]
                
                assert torch.argmax(placeholder_one_hot) == PLACEHOLDER_TOKEN_ID
                
                placeholder_onehot_sum = placeholder_one_hot.sum()
                
                assert placeholder_onehot_sum == 1
                
                adversarial_step_rate = 1.0
                
                placeholder_one_hot += adversarial_step_rate * grad
                
                assert placeholder_onehot_sum != placeholder_one_hot.sum()
                
                top_candidates_one_sample = torch.topk(placeholder_one_hot, 5).indices.tolist()
                
                if PLACEHOLDER_TOKEN_ID in top_candidates_one_sample:
                    top_candidates_one_sample.remove(PLACEHOLDER_TOKEN_ID)
                    
                top_candidates.append(top_candidates_one_sample)
            
            optimizer.zero_grad()
            model.eval()
            
            del(batch["inputs_embeds"])
            del(batch["position_ids"])
            
            predicted_classes_adversarial = predicted_classes_original[:]
            batch["input_ids"] = original_input_ids.to(device)
            
            with torch.no_grad():
                
                for k in range(4):
                    
                    for u, candidate_set in enumerate(top_candidates):
                        
                        if predicted_classes_adversarial[u] == predicted_classes_original[u]:
                        
                            batch["input_ids"][u, placeholder_positions[u]] = candidate_set[k]
                        
                    outputs = model(**batch)
                    
                    logits = outputs.logits
            
                    predicted_classes_adversarial = torch.argmax(logits, dim=-1).tolist()
                    
                    if [x + y for x, y in zip(predicted_classes_adversarial, predicted_classes_original)] == [1, 1, 1, 1]:
                        break
                        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch["labels"])
    
        loss.backward()
        
        logits = outputs.logits
        
        losses += [loss.item()]
        predictions += torch.argmax(logits, dim=-1).tolist()
        labels += batch["labels"].tolist()

        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)
    
    model.eval()

    for k, TF_TO_APPLY_ON_TEST in enumerate(semantic_preserving_transformations):
        
        if machine == "cluster":
            m3=mydata.iloc[list(test_index)]
        else:
            m3=mydata.iloc[list(test_index)[:8]]
        
        m3 = apply_transformation(m3, semantic_preserving_transformations, TF_TO_APPLY_ON_TEST, training_set_sample_neg, training_set_sample_pos, TF_TO_APPLY_ON_TEST)

        test_encodings, test_labels = encodeDataframe(m3)

        test_dataset = MyCustomDataset(test_encodings, test_labels)

        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4)
        
        predictions = []
        labels = []
    
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if random_guessing:
                predictions += len(batch["labels"].tolist()) * [random.randint(0, 1)]
                labels += batch["labels"].tolist()
            else:
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                
                predictions += torch.argmax(logits, dim=-1).tolist()
                labels += batch["labels"].tolist()
            
            progress_bar.update(1)
        
        test_accuracy = accuracy_score(labels, predictions)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
        test_tn, test_fp, test_fn, test_tp = confusion_matrix(labels, predictions).ravel()
        
        if epoch == 0:
            results[TF_TO_APPLY_ON_TEST.__name__] = dict()
        results[TF_TO_APPLY_ON_TEST.__name__][epoch] = dict()
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["epoch"] = epoch
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["train/loss"] = np.array(losses).mean()
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/accuracy"] = test_accuracy
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/precision"] = test_precision
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/recall"] = test_recall
        results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/f1"] = test_f1
        
    predictions_fixed = []
    labels_fixed = []

    for batch in fixed_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if random_guessing:
            predictions_fixed += len(batch["labels"].tolist()) * [random.randint(0, 1)]
            labels_fixed += batch["labels"].tolist()
        else:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            
            predictions_fixed += torch.argmax(logits, dim=-1).tolist()
            labels_fixed += batch["labels"].tolist()
        
        progress_bar.update(1)
        
    assert 0 == np.sum(np.array(labels_fixed))
    
    predictions_not_fixed = []
    labels_not_fixed = []

    for batch in not_fixed_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if random_guessing:
            predictions_not_fixed += len(batch["labels"].tolist()) * [random.randint(0, 1)]
            labels_not_fixed += batch["labels"].tolist()
        else:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            
            predictions_not_fixed += torch.argmax(logits, dim=-1).tolist()
            labels_not_fixed += batch["labels"].tolist()
        
        progress_bar.update(1)
        
    assert len(labels_not_fixed) == np.sum(np.array(labels_not_fixed))
    
    predictions = predictions_fixed + predictions_not_fixed
    labels = labels_fixed + labels_not_fixed
    
    test_accuracy = accuracy_score(labels, predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(labels, predictions).ravel()
    
    switch = np.sum(np.array(predictions_fixed) != np.array(predictions_not_fixed)) / len(predictions_not_fixed)
    
    if epoch == 0:
        results["fixed_nonfixed"] = dict()
    results["fixed_nonfixed"][epoch] = dict()
    results["fixed_nonfixed"][epoch]["epoch"] = epoch
    results["fixed_nonfixed"][epoch]["train/loss"] = np.array(losses).mean()
    results["fixed_nonfixed"][epoch]["test/accuracy"] = test_accuracy
    results["fixed_nonfixed"][epoch]["test/precision"] = test_precision
    results["fixed_nonfixed"][epoch]["test/recall"] = test_recall
    results["fixed_nonfixed"][epoch]["test/f1"] = test_f1
    results["fixed_nonfixed"][epoch]["test/switch"] = switch
        

if random_guessing:
    save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "-RG"
else:
    save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "-AdversarialTraining"
    
with open('./' + pathprefix + 'results/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(results, f)

