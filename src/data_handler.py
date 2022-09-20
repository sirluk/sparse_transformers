import os
import pickle
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoConfig

from typing import Union, Tuple, List


class TextDS(Dataset):
    def __init__(self, text, tokenizer, max_length):
        self.text = text
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int):
        tokenized_sample = self.tokenizer(
            self.text[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        token_ids_sample = tokenized_sample['input_ids']
        token_type_ids_sample = tokenized_sample['token_type_ids']
        attention_masks_sample = tokenized_sample['attention_mask']
        return token_ids_sample, token_type_ids_sample, attention_masks_sample


def multiprocess_tokenization(text_list, tokenizer, max_length, num_workers=16):
    ds = TextDS(text_list, tokenizer, max_length)
    _loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=num_workers, drop_last=False)
    token_ids = []
    token_type_ids = []
    attention_masks = []
    for tokenized_batch, token_type_ids_batch, attention_masks_batch in _loader:
        token_ids.append(tokenized_batch)
        token_type_ids.append(token_type_ids_batch)
        attention_masks.append(attention_masks_batch)

    token_ids = torch.cat(token_ids, dim=0).squeeze(1)
    token_type_ids = torch.cat(token_type_ids, dim=0).squeeze(1)
    attention_masks = torch.cat(attention_masks, dim=0).squeeze(1)

    return token_ids, token_type_ids, attention_masks


def read_label_file(filepath):
    with open(filepath) as f:
        data = f.read()
        return {v:k for k,v in enumerate([l for l in data.split("\n") if len(l)>0])}


def get_data_loader(
    task_key: str,
    protected_key: Union[str, list],
    text_key: str,
    tokenizer: Tokenizer,
    data_path: Union[str, os.PathLike],
    labels_task_path: Union[str, os.PathLike],
    labels_prot_path: Union[None, str, os.PathLike, list] = None,
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = True,
    debug: bool = False
):

    def batch_fn(batch):
        input_ids, token_type_ids, attention_masks, labels_task = [torch.stack(l) for l in zip(*batch)]
        x = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_masks
        }
        return x, labels_task


    def batch_fn_prot(batch):
        # b = input_ids, token_type_ids, attention_masks, labels_task, labels_prot1, labels_prot2, ...
        b = [torch.stack(l) for l in zip(*batch)]
        x = {
            "input_ids": b[0],
            "token_type_ids": b[1],
            "attention_mask": b[2]
        }
        return x, *b[3:]


    if isinstance(protected_key, str):
        protected_key = [protected_key]
        labels_prot_path = [labels_prot_path]

    with open(data_path, 'rb') as file:
        data_dicts = pickle.load(file)

    if debug:
        cutoff = min(int(batch_size*10), len(data_dicts))
        data_dicts = data_dicts[:cutoff]

    keys = [task_key, *protected_key, text_key]
    x = [[d[k] for k in keys] for d in data_dicts]

    data = dict(zip(keys, zip(*x)))

    input_ids, token_type_ids, attention_masks = multiprocess_tokenization(data[text_key], tokenizer, max_length)

    labels_task = read_label_file(labels_task_path)
    labels_task = torch.tensor([labels_task[str(t)] for t in data[task_key]], dtype=torch.long)

    tds = [
        input_ids,
        token_type_ids,
        attention_masks,
        labels_task
    ]

    if labels_prot_path is not None:
        for k, f in zip(protected_key, labels_prot_path):
            labels_prot = read_label_file(f)
            tds.append(torch.tensor([labels_prot[str(t)] for t in data[k]], dtype=torch.long))
            collate_fn = batch_fn_prot
    else:
        collate_fn = batch_fn

    _dataset = TensorDataset(*tds)

    _loader = DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=collate_fn)

    return _loader


def get_num_labels(label_file: Union[str, os.PathLike]) -> int:
    num_labels = len(read_label_file(label_file))
    return 1 if num_labels==2 else num_labels


def get_max_length(data_paths: List[Union[str, os.PathLike]], text_key: str):
    
    data_dicts = []
    for p in data_paths:
        with open(p, 'rb') as file:
            data_dicts.extend(pickle.load(file))

    texts = [d[text_key] for d in data_dicts]
    return max([len(x) for x in texts])


def get_data(
    args_train: argparse.Namespace,
    attr_idx: int = 0,
    use_all_attr: bool = False,
    debug: bool = False
) -> Tuple[DataLoader, DataLoader, int, int]:
    
    num_labels = get_num_labels(args_train.labels_task_path)

    if isinstance(args_train.labels_protected_path, list):
        if use_all_attr:
            num_labels_protected = [get_num_labels(x) for x in args_train.labels_protected_path]
        else:
            setattr(args_train, "labels_protected_path", args_train.labels_protected_path[attr_idx])
            num_labels_protected = [get_num_labels(args_train.labels_protected_path)]
    else:
        num_labels_protected = [get_num_labels(args_train.labels_protected_path)]

    if isinstance(args_train.protected_key, list) and not use_all_attr:
        setattr(args_train, "protected_key", args_train.protected_key[attr_idx])

    cfg = AutoConfig.from_pretrained(args_train.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)

    length_check = [
        get_max_length([args_train.train_pkl, args_train.val_pkl], args_train.text_key),
        cfg.max_position_embeddings
    ]
    if args_train.tokenizer_max_length is not None:
        length_check.append(args_train.tokenizer_max_length)
    max_length = min(length_check)

    train_loader = get_data_loader(
        task_key = args_train.task_key,
        protected_key = args_train.protected_key,
        text_key = args_train.text_key,
        tokenizer = tokenizer,
        data_path = args_train.train_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = args_train.labels_protected_path,
        batch_size = args_train.batch_size,
        max_length = max_length,
        debug = debug
    )
    val_loader = get_data_loader(
        task_key = args_train.task_key,
        protected_key = args_train.protected_key,
        text_key = args_train.text_key,
        tokenizer = tokenizer,
        data_path = args_train.val_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = args_train.labels_protected_path,
        batch_size = args_train.batch_size,
        max_length = max_length,
        shuffle = False,
        debug = debug
    )
    return train_loader, val_loader, num_labels, *num_labels_protected