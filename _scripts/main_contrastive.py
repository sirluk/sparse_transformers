import pickle
import argparse
import ruamel.yaml as yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from transformers import AutoTokenizer

from src.data_handler import multiprocess_tokenization, read_label_file, get_data_loader
from src.training_logger import TrainLogger
from src.models.model_contrastive import ContModel
from src.adv_attack import adv_attack
from src.utils import (
    get_device,
    set_num_epochs_debug,
    get_callables,
    get_num_labels
)

from typing import Tuple


torch.manual_seed(0)


def get_data_loader_pairs(
    tokenizer,
    data_path,
    labels_task_path,
    labels_prot_path,
    batch_size=16,
    max_length=200,
    raw=False,
    shuffle=True,
    debug=False
):

    def batch_fn_pairs(batch):
        b = [torch.stack(l) for l in zip(*batch)]
        x1 = {
            "input_ids": b[0],
            "token_type_ids": b[1],
            "attention_mask": b[2]
        }
        x2 = {
            "input_ids": b[5],
            "token_type_ids": b[6],
            "attention_mask": b[7]
        }
        return x1, b[3], b[4], x2, b[8], b[9]

    if raw:
        text_fn = lambda x: x['raw'][x['start_pos']:]
    else:
        text_fn = lambda x: x["bio"]

    with open(data_path, 'rb') as file:
        data, id_pairs = pickle.load(file)

    if debug:
        cutoff = min(int(batch_size*10), len(id_pairs))
        id_pairs = id_pairs[:cutoff]

    keys = ["gender", "title"]
    x = [[d[k] for k in keys] + [text_fn(d)] for d in data]
    keys.append("text")

    data_dict = dict(zip(keys, zip(*x)))

    input_ids, token_type_ids, attention_masks = multiprocess_tokenization(list(data_dict["text"]), tokenizer, max_length)

    labels_task = read_label_file(labels_task_path)
    labels_task = torch.tensor([labels_task[t] for t in data_dict["title"]], dtype=torch.long)
    labels_prot = read_label_file(labels_prot_path)
    labels_prot = torch.tensor([labels_prot[t] for t in data_dict["gender"]], dtype=torch.long)

    tokenized_samples = []
    for id1, id2 in id_pairs:
        tokenized_samples.append((
            input_ids[id1],
            token_type_ids[id1],
            attention_masks[id1],
            labels_task[id1],
            labels_prot[id1],
            input_ids[id2],
            token_type_ids[id2],
            attention_masks[id2],
            labels_task[id2],
            labels_prot[id2],
        ))
    tds = [torch.stack(t) for t in zip(*tokenized_samples)]

    _dataset = TensorDataset(*tds)

    _loader = DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=batch_fn_pairs)

    return _loader


def get_data_loader_triplets(
    tokenizer,
    data_path,
    labels_task_path,
    labels_prot_path,
    batch_size=16,
    max_length=200,
    raw=False,
    shuffle=True,
    debug=False
):

    def batch_fn_triplets(batch):
        def get_encoded_text_dict(input_ids, token_type_ids, attention_mask):
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask
            }
        b = [torch.stack(l) for l in zip(*batch)]
        x1 = get_encoded_text_dict(*b[:3])
        x2 = get_encoded_text_dict(*b[5:8])
        x3 = get_encoded_text_dict(*b[10:13])
        return x1, b[3], b[4], x2, b[8], b[9], x3, b[13], b[14]

    if raw:
        text_fn = lambda x: x['raw'][x['start_pos']:]
    else:
        text_fn = lambda x: x["bio"]

    with open(data_path, 'rb') as file:
        data, id_triplets = pickle.load(file)

    if debug:
        cutoff = min(int(batch_size*10), len(id_triplets))
        id_triplets = id_triplets[:cutoff]

    keys = ["gender", "title"]
    x = [[d[k] for k in keys] + [text_fn(d)] for d in data]
    keys.append("text")

    data_dict = dict(zip(keys, zip(*x)))

    input_ids, token_type_ids, attention_masks = multiprocess_tokenization(list(data_dict["text"]), tokenizer, max_length)

    labels_task = read_label_file(labels_task_path)
    labels_task = torch.tensor([labels_task[t] for t in data_dict["title"]], dtype=torch.long)
    labels_prot = read_label_file(labels_prot_path)
    labels_prot = torch.tensor([labels_prot[t] for t in data_dict["gender"]], dtype=torch.long)

    tokenized_samples = []
    for id1, id2, id3 in id_triplets:
        tokenized_samples.append((
            input_ids[id1],
            token_type_ids[id1],
            attention_masks[id1],
            labels_task[id1],
            labels_prot[id1],
            input_ids[id2],
            token_type_ids[id2],
            attention_masks[id2],
            labels_task[id2],
            labels_prot[id2],
            input_ids[id3],
            token_type_ids[id3],
            attention_masks[id3],
            labels_task[id3],
            labels_prot[id3],
        ))
    tds = [torch.stack(t) for t in zip(*tokenized_samples)]

    _dataset = TensorDataset(*tds)

    _loader = DataLoader(_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=False, collate_fn=batch_fn_triplets)

    return _loader


def get_data_contrastive(args_train: argparse.Namespace, debug: bool = False) -> Tuple[DataLoader, DataLoader, int, int]:
    num_labels = get_num_labels(args_train.labels_task_path)
    num_labels_protected = get_num_labels(args_train.labels_protected_path)
    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)
    train_loader = get_data_loader_triplets(
        tokenizer = tokenizer,
        data_path = "../data/train_triplets_extended.pkl", # args_train.train_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = args_train.labels_protected_path,
        batch_size = args_train.batch_size,
        max_length = 200,
        raw = False,
        debug = debug
    )
    val_loader = get_data_loader(
        tokenizer = tokenizer,
        data_path = args_train.val_pkl,
        labels_task_path = args_train.labels_task_path,
        labels_prot_path = args_train.labels_protected_path,
        batch_size = args_train.batch_size,
        max_length = 200,
        raw = False,
        shuffle = False,
        debug = debug
    )
    return train_loader, val_loader, num_labels, num_labels_protected


def train_contrastive(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train):

    loss_fn, pred_fn, metrics = get_callables(num_labels)
    loss_fn_protected, pred_fn_protected, metrics_protected = get_callables(num_labels_protected)

    trainer = ContModel(
        model_name = args_train.model_name,
        num_labels_task = num_labels,
        num_labels_protected = num_labels_protected,
        task_dropout = args_train.task_dropout,
        task_n_hidden = args_train.task_n_hidden,
        adv_dropout = args_train.adv_dropout,
        adv_n_hidden = args_train.adv_n_hidden,
        adv_count = args_train.adv_count,
        bottleneck = args_train.bottleneck,
        bottleneck_dim = args_train.bottleneck_dim,
        bottleneck_dropout = args_train.bottleneck_dropout
    )
    trainer.to(device)
    trainer_cp = trainer.fit(
        train_loader = train_loader,
        val_loader = val_loader,
        logger = train_logger,
        loss_fn = loss_fn,
        pred_fn = pred_fn,
        metrics = metrics,
        loss_fn_protected = loss_fn_protected,
        pred_fn_protected = pred_fn_protected,
        metrics_protected = metrics_protected,
        num_epochs = args_train.num_epochs,
        learning_rate = args_train.learning_rate,
        learning_rate_bottleneck = args_train.learning_rate_bottleneck,
        learning_rate_task_head = args_train.learning_rate_task_head,
        learning_rate_adv_head = args_train.learning_rate_adv_head,
        optimizer_warmup_steps = args_train.optimizer_warmup_steps,
        max_grad_norm = args_train.max_grad_norm,
        output_dir = args_train.output_dir
    )
    trainer = ContModel.load_checkpoint(trainer_cp)
    trainer.to(device)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--run_adv_attack", type=bool, default=True, help="Set to false if you do not want to run adverserial attack after training")
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    base_args = parser.parse_args()

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    # train_cfg = "train_config_baseline" if base_args.baseline else "train_config_diff_pruning"
    args_train = argparse.Namespace(**cfg["train_config_baseline"], **cfg["data_config"], **cfg["model_config"])
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    if base_args.debug:
        args_train = set_num_epochs_debug(args_train)
        args_attack = set_num_epochs_debug(args_attack)

    device = get_device(base_args.gpu_id)

    train_loader, val_loader, num_labels, num_labels_protected = get_data_contrastive(args_train, debug=base_args.debug)

    logger_name = "_".join([
        "DEBUG_contrastive" if base_args.debug else "contrastive",
        str(args_train.batch_size),
        str(args_train.learning_rate)
    ])
    train_logger = TrainLogger(
        log_dir = Path("logs"),
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )

    print(f"Running {train_logger.logger_name}")

    trainer = train_contrastive(device, train_loader, val_loader, num_labels, num_labels_protected, train_logger, args_train)

    if base_args.run_adv_attack:
        loss_fn, pred_fn, metrics = get_callables(num_labels_protected)
        adv_attack(
            trainer = trainer,
            train_loader = train_loader,
            val_loader = val_loader,
            logger = train_logger,
            loss_fn = loss_fn,
            pred_fn = pred_fn,
            metrics = metrics,
            num_labels = num_labels_protected,
            adv_n_hidden = args_attack.adv_n_hidden,
            adv_count = args_attack.adv_count,
            adv_dropout = args_attack.adv_dropout,
            num_epochs = args_attack.num_epochs,
            lr = args_attack.learning_rate
        )


if __name__ == "__main__":

    main()

