import os
import json
import copy
from tqdm import tqdm
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from torch.utils.data import DataLoader, Dataset
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from UIE.utils import TYPE_START, TYPE_END, TEXT_START, SPAN_START
from UIE.utils import T5BertTokenizer
from transformers import DataCollatorForSeq2Seq
import pandas as pd 
import random

class UIEDataset(Dataset):
    def __init__(self, args, tokenizer, is_training=False):
        super().__init__()
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.prefix = ""
        self.args = args
        self.max_target_length = args.max_target_length
        if is_training:
            filenames = args.train_file
        else:
            filenames = args.val_file
        data = pd.read_csv(filenames)
        self.examples = data.to_dict(orient="records")
        with open("data/sentiment.json",'r') as f:
            self.sentiment = json.load(f)
        
    def __len__(self):
        return len(self.examples)

    def add_schema(self,target):
        target_str = TYPE_START
        # random.shuffle(target)
        for line in target:
            label = line['label'][-1]
            sentiment=self.sentiment[str(line['sentiment'])]
            # target_str +=TYPE_START+"类别"+SPAN_START+label+TYPE_START+"情感"+SPAN_START+sentiment+TYPE_END+TYPE_END
            target_str +=TYPE_START+label.strip()+SPAN_START+sentiment.strip()+TYPE_END

        target_str+=TYPE_END
        return target_str

    def __getitem__(self, index):
        example = self.examples[index]
        inputs = self.prefix + example['text']
        targets = eval(example["result"])
        targets = self.add_schema(targets)
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_source_length, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


class CollateFn():
    def __init__(self, args, tokenizer, model) -> None:
        args = args
        model = model
        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if args.float16 else None,
        )

    def __call__(self, features):
        return self.data_collator(features)


def add_special_token_tokenizer(pretrain_path):
    """为tokenizer中添加特殊符号
    Args:
        pretrain_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    if 'char' in pretrain_path:
        tokenizer = T5BertTokenizer.from_pretrained(pretrain_path,cache_dir="./T5")
    else:
        tokenizer = T5Tokenizer.from_pretrained(pretrain_path,cache_dir="./T5")
    to_add_special_token = list()
    for special_token in [TYPE_START, TYPE_END, TEXT_START, SPAN_START]:
        if special_token not in tokenizer.get_vocab():
            to_add_special_token += [special_token]
    # tokenizer.add_tokens(tokenizer.additional_special_tokens + to_add_special_token, special_tokens=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": tokenizer.special_tokens_map_extended['additional_special_tokens'] + to_add_special_token})

    return tokenizer
