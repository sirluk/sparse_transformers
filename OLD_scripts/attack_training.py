import ruamel.yaml as yaml
import argparse
import math
from pathlib import Path
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model_heads import AdvHead
from src.model import DiffNetwork
from src.model_adverserial import AdverserialNetwork
from src.model_task import TaskNetwork
from src.training_logger import TrainLogger
from src.utils import get_metrics, get_loss_fn, get_num_labels, dict_to_device
from src.data_handler import get_data_loader
from src.adv_attack import adv_attack, evaulate_adv_attack

from typing import Callable, Dict, Union

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(base_args, args):

    if "task" in args.checkpoint_path:
        model = TaskNetwork.load_checkpoint(args.checkpoint_path)
    elif "baseline" in args.checkpoint_path:
        model = AdverserialNetwork.load_checkpoint(args.checkpoint_path)
    else:
        model = DiffNetwork.load_checkpoint(args.checkpoint_path, remove_parametrizations=True)

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    
    num_labels = get_num_labels(args.labels_protected_path)

    emb_dim = model.encoder.embeddings.word_embeddings.embedding_dim
    adv_head = AdvHead(args.adv_count, hid_sizes=emb_dim, num_labels=num_labels, dropout_prob=args.adv_dropout)
    
    model.to(DEVICE)
    adv_head.to(DEVICE)

    train_loader = get_data_loader(
        tokenizer=tokenizer,
        data_path=args.train_pkl, 
        labels_task_path=args.labels_task_path,
        labels_prot_path=args.labels_protected_path,
        batch_size=args.batch_size, 
        max_length = 200,
        raw=base_args.raw,
        debug=base_args.debug
    )
    
    eval_loader = get_data_loader(
        tokenizer=tokenizer,
        data_path=args.val_pkl, 
        labels_task_path=args.labels_task_path,
        labels_prot_path=args.labels_protected_path,
        batch_size=args.batch_size, 
        max_length = 200,
        raw=base_args.raw,
        shuffle=False,
        debug=base_args.debug
    )

    logger_name = "_".join([
        "adv_attack",
        args.checkpoint_path.split("/")[-1]
    ])
    train_logger = TrainLogger(
        log_dir = Path("logs") / "adv_attack",
        logger_name = logger_name,
        logging_step = args.logging_step
    )            
    
    adv_attack(
        encoder = model.encoder,
        adv_head = adv_head,
        train_loader = train_loader,
        val_loader = eval_loader,
        logger = train_logger,
        loss_fn = get_loss_fn(num_labels),
        metrics = get_metrics(num_labels),
        num_epochs = args.num_epochs,
        lr = args.learning_rate
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--raw", type=bool, default=False, help="")
    base_args = parser.parse_args()
    
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)    
    args = argparse.Namespace(**{**cfg["adv_attack"], **cfg["data_config"]})

    main(base_args, args)
