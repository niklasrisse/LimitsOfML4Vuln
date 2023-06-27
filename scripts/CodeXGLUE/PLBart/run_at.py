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
from transformers import get_scheduler

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

my_tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-c-cpp-defect-detection")


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
    
    encodings = my_tokenizer.batch_encode_plus(df.func, max_length=1024, padding='max_length', truncation=True)
    
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

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class PLBartForSequenceClassificationWithEmbeds(PLBartForSequenceClassification):
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        eos_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        if eos_mask is None:
            eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
        
def placeholder_injection(code):
    
    begin_of_function = code.index('{')

    code = code[0:begin_of_function + 1] + "\nchar placeholder;\n" + code[begin_of_function + 1:]
    
    return code
    
    

    
TF_TO_APPLY_ON_TRAIN = placeholder_injection
PLACEHOLDER_TOKEN_ID = 5070
SPECIAL_TOKENS_IDS = [0, 1, 2, 3]

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

TRAINING_BATCH_SIZE = 8

train_encodings, train_labels = encodeDataframe(m1)
test_encodings, test_labels = encodeDataframe(m3)
fixed_encodings, fixed_labels = encodeDataframe(m4_changed)
not_fixed_encodings, not_fixed_labels = encodeDataframe(m4)

train_dataset = MyCustomDataset(train_encodings, train_labels)
test_dataset = MyCustomDataset(test_encodings, test_labels)
fixed_dataset = MyCustomDataset(fixed_encodings, fixed_labels)
not_fixed_dataset = MyCustomDataset(not_fixed_encodings, not_fixed_labels)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAINING_BATCH_SIZE)
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

model = PLBartForSequenceClassificationWithEmbeds.from_pretrained("uclanlp/plbart-base")
model.to(device)

optimizer = AdamW(model.parameters(), lr=3e-6)
lr_scheduler = get_scheduler(
     name="linear", optimizer=optimizer, num_warmup_steps=2500, num_training_steps=num_training_steps
)

criterion = torch.nn.CrossEntropyLoss() 
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
                
        if len(placeholder_positions) == 8:
            
            optimizer.zero_grad()
            
            embedding_weights = model.model.encoder.embed_tokens.weight
            input_ids_onehot = torch.nn.functional.one_hot(batch['input_ids'], num_classes=50005).to(torch.float)
            input_ids_onehot.requires_grad = True
            
            batch['eos_mask'] = batch['input_ids'].eq(model.config.eos_token_id)
            batch['decoder_input_ids'] = shift_tokens_right(batch['input_ids'], model.model.config.pad_token_id)
            input_embeds = input_ids_onehot @ embedding_weights * model.model.encoder.embed_scale
            
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
            del(batch["decoder_input_ids"])
            del(batch["eos_mask"])
            
            predicted_classes_adversarial = predicted_classes_original[:]
            batch["input_ids"] = original_input_ids.to(device)
            
            with torch.no_grad():
                
                for k in range(4):
                    
                    for u, candidate_set in enumerate(top_candidates):
                        
                        if predicted_classes_adversarial[u] == predicted_classes_original[u]:
                            
                            if candidate_set[k] not in SPECIAL_TOKENS_IDS:
                                batch["input_ids"][u, placeholder_positions[u]] = candidate_set[k]
                        
                    outputs = model(**batch)
                    
                    logits = outputs.logits
            
                    predicted_classes_adversarial = torch.argmax(logits, dim=-1).tolist()
                    
                    if [x + y for x, y in zip(predicted_classes_adversarial, predicted_classes_original)] == [1, 1, 1, 1, 1, 1, 1, 1]:
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

