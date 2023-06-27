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
parser.add_argument("--trafo", required=True)
parser.add_argument("--random", default=False, required=False)
args = parser.parse_args()
trafo_from_args = STR_TO_TRAFO[args.trafo]
random_guessing = args.random == "True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print(device)

# deterministic settings for exact reproducibility
seedlist = [42, 834, 692, 489, 901, 408, 819, 808, 531, 166]
fixed_seed = 834 #random.choice(seedlist)

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


mydataset="vuldeepecker"

m1 = pd.read_pickle("./" + pathprefix + 'datasets/VulDeePecker/VulDeePecker-Train.pkl')
m3 = pd.read_pickle("./" + pathprefix + 'datasets/VulDeePecker/VulDeePecker-Test.pkl')



m1 = m1.rename(columns={'functionSource': 'func'})
m3 = m3.rename(columns={'functionSource': 'func'})

m1 = m1.rename(columns={'label': 'target'})
m3 = m3.rename(columns={'label': 'target'})

m1["func"] = m1["func"].astype(str)
m3["func"] = m3["func"].astype(str)

print("Before Pre-Processing")
print(m1.shape[0])
print(m3.shape[0])

m3_func_list = m3['func'].to_list()
m1_func_list = m1['func'].to_list()

counter = 0
num_duplicates = 0

funcs_to_drop = []

for m1_func in m1_func_list:
    counter +=1
    if m1_func in m3_func_list and m1_func not in funcs_to_drop:
        num_duplicates += 1
        
        funcs_to_drop.append(m1_func)
        

for func in funcs_to_drop:
    m3 = m3.drop(m3[m3["func"] == func].index)
    m1 = m1.drop(m1[m1["func"] == func].index)

print("After removal of inter set duplicates")
print(m1.shape[0])
print(m3.shape[0])

m3_func_list = m3['func'].to_list()
funcs_to_drop = []

for m3_func in m3_func_list:
    
    if m3_func_list.count(m3_func) > 1 and m3_func not in funcs_to_drop:
        funcs_to_drop.append(m3_func)
        
for func in funcs_to_drop:
    # m3 = m3.drop(m3[m3["func"] == func].index)
    m3 = m3.drop(m3[m3["func"] == func].index)
        
print("After removal of m3 intra set duplicates")
print(m1.shape[0])
print(m3.shape[0])

m1_func_list = m1['func'].to_list()
funcs_to_drop = []

for m1_func in m1_func_list:
    
    if m1_func_list.count(m1_func) > 1 and m1_func not in funcs_to_drop:
        funcs_to_drop.append(m1_func)
        
for func in funcs_to_drop:
    # m3 = m3.drop(m3[m3["func"] == func].index)
    m1 = m1.drop(m1[m1["func"] == func].index)
        
print("After removal of m1 intra set duplicates")
print(m1.shape[0])
print(m3.shape[0])

m1_func_list = m1['func'].to_list()
funcs_to_drop = []

for m1_func in m1_func_list:
    
    if  "Calling good()..." in m1_func or "Calling bad()..." in m1_func:
        funcs_to_drop.append(m1_func)
        
for func in funcs_to_drop:
    # m3 = m3.drop(m3[m3["func"] == func].index)
    m1 = m1.drop(m1[m1["func"] == func].index)
        
print("After removal of calling good and calling bad from m1")
print(m1.shape[0])
print(m3.shape[0])

m3_func_list = m3['func'].to_list()
funcs_to_drop = []

for m3_func in m3_func_list:
    
    if "Calling good()..." in m3_func or "Calling bad()..." in m3_func:
        funcs_to_drop.append(m3_func)
        
for func in funcs_to_drop:
    # m3 = m3.drop(m3[m3["func"] == func].index)
    m3 = m3.drop(m3[m3["func"] == func].index)
        
print("After removal of calling good and calling bad from m3")
print(m1.shape[0])
print(m3.shape[0])

print(m3[m3["target"] == 1].iloc[9]["func"])


def random_two_letters(length = 2):
    
    letters = string.ascii_lowercase
    random_letters = ''.join(random.choice(letters) for i in range(length))
    
    return random_letters

def custom_cleaning_function(code):
    ## Remove code comments
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat,'',code)
    
    code = code.replace("good", random_two_letters())
    code = code.replace("bad", random_two_letters())
    code = code.replace("Good", random_two_letters())
    code = code.replace("Bad", random_two_letters())
    code = code.replace("B2G", random_two_letters())
    code = code.replace("G2B", random_two_letters())
    code = code.replace("B2G2", random_two_letters())
    code = code.replace("G2B2", random_two_letters())
    code = code.replace("source", random_two_letters())
    code = code.replace("Source", random_two_letters())
    code = code.replace("data", random_two_letters())
    code = code.replace("Sink", random_two_letters())
    code = code.replace("sink", random_two_letters())
    
    words = code.split(" ")
    
    for word in words:
        if word.count("_") > 5:
            if "(" in word:
                word = word.split("(")[0]
                
            if ";" in word:
                word = word.split(";")[0]
            
            if word.count("_") > 5 and ")" not in word and "(" not in word and "," not in word and ":" not in word and "\n" not in word and "\t" not in word and ";" not in word and ">" not in word and "*" not in word and "}" not in word:
                
                code = code.replace(word, random_two_letters())
    
    return code

m1.func = m1.func.apply(custom_cleaning_function)
m3.func = m3.func.apply(custom_cleaning_function)

print(m3[m3["target"] == 1].iloc[10]["func"])


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

    
TF_TO_APPLY_ON_TRAIN = trafo_from_args

semantic_preserving_transformations = [no_transformation, tf_1, tf_2, tf_3, tf_4, tf_5, tf_6, tf_7, tf_8, tf_12, tf_11, tf_10, tf_13]

m1_not_vulnerable = m1[m1["target"] == 0]
training_set_sample_neg = m1_not_vulnerable.iloc[0]["func"]

m1_vulnerable = m1[m1["target"] == 1]
training_set_sample_pos = m1_vulnerable.iloc[0]["func"]
    
m1 = apply_transformation(m1, semantic_preserving_transformations, TF_TO_APPLY_ON_TRAIN, training_set_sample_neg, training_set_sample_pos, None)


TRAINING_BATCH_SIZE = 4

train_encodings, train_labels = encodeDataframe(m1)
test_encodings, test_labels = encodeDataframe(m3)

train_dataset = MyCustomDataset(train_encodings, train_labels)
test_dataset = MyCustomDataset(test_encodings, test_labels)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAINING_BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4)

print(len(train_dataloader))
print(len(test_dataloader))

num_epochs = 8
num_training_steps = num_epochs * len(train_dataloader)
num_testing_steps = num_epochs * len(test_dataloader)

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

progress_bar = tqdm(range(num_training_steps + num_testing_steps * len(semantic_preserving_transformations)))

for epoch in range(num_epochs):
    
    model.train()
    
    losses = []
    predictions = []
    labels = []
    
    for batch in train_dataloader:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if not random_guessing:
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
        
        m3_transformed = m3.copy(deep=True)
        
        m3_transformed = apply_transformation(m3_transformed, semantic_preserving_transformations, TF_TO_APPLY_ON_TEST, training_set_sample_neg, training_set_sample_pos, TF_TO_APPLY_ON_TEST)

        test_encodings, test_labels = encodeDataframe(m3_transformed)

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
        

if random_guessing:
    save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "-RG"
else:
    save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "-" + TF_TO_APPLY_ON_TRAIN.__name__ + "VDP"
    
with open('./' + pathprefix + 'results/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(results, f)

