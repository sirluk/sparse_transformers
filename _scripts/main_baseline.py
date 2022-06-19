# import IPython; IPython.embed(); exit(1)

import ruamel.yaml as yaml
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data_handler import get_data_loader 
from src.model import DiffNetwork
from src.model_adverserial import AdverserialNetwork
from src.model_task import TaskNetwork
from src.model_heads import AdvHead
from src.training_logger import TrainLogger
from src.utils import get_metrics, get_loss_fn, get_num_labels
from src.adv_attack import adv_attack

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(base_args, args_train, args_attack):    
    
    Path(args_train.output_dir).mkdir(parents=True, exist_ok=True)

    num_labels = get_num_labels(args_train.labels_task_path)
    num_labels_protected = get_num_labels(args_train.labels_protected_path)

    tokenizer = AutoTokenizer.from_pretrained(args_train.model_name)

    train_loader = get_data_loader(
        tokenizer=tokenizer,
        data_path=args_train.train_pkl, 
        labels_task_path=args_train.labels_task_path,
        labels_prot_path=args_train.labels_protected_path,
        batch_size=args_train.batch_size, 
        max_length = 200,
        raw=base_args.raw,
        debug=base_args.debug
    )
    
    eval_loader = get_data_loader(
        tokenizer=tokenizer,
        data_path=args_train.val_pkl, 
        labels_task_path=args_train.labels_task_path,
        labels_prot_path=args_train.labels_protected_path,
        batch_size=args_train.batch_size, 
        max_length = 200,
        raw=base_args.raw,
        shuffle=False,
        debug=base_args.debug
    )

    method = "baseline_adverserial" if args_train.adverserial else "baseline_task"
    if args_train.bitfit: method = method + "_bitfit"
    logger_name = "_".join([
        method,
        args_train.model_name.split('/')[-1],
        str(args_train.batch_size),
        str(args_train.learning_rate),
        str(args_train.adv_rev_ratio)
    ])
    train_logger = TrainLogger(
        log_dir = Path("logs"),
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )

    if args_train.adverserial:
        trainer = AdverserialNetwork(
            args_train.model_name,
            num_labels,
            num_labels_protected,
            adv_count=args_train.adv_count,
            adv_rev_ratio=args_train.adv_rev_ratio
        )  
    else:
        trainer = TaskNetwork(
            args_train.model_name,
            num_labels
        ) 
        
    trainer.to(DEVICE)
        
    if args_train.adverserial:
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            get_loss_fn(num_labels_protected),
            get_metrics(num_labels_protected),
            args_train.num_epochs,
            args_train.learning_rate,
            args_train.learning_rate_adverserial,
            args_train.warmup_steps,
            args_train.max_grad_norm,
            args_train.output_dir,
            args_train.bitfit
        )
    else:    
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            args_train.num_epochs,
            args_train.learning_rate,
            args_train.warmup_steps,
            args_train.max_grad_norm,
            args_train.output_dir,
            args_train.bitfit
        )
        
    adv_head = AdvHead(
        args_attack.adv_count,
        hid_sizes=trainer.encoder.embeddings.word_embeddings.embedding_dim,
        num_labels=num_labels_protected,
        dropout_prob=args_attack.adv_dropout
    )
    adv_head.to(DEVICE)   
    
    adv_attack(
        encoder = trainer.encoder,
        adv_head = adv_head,
        train_loader = train_loader,
        val_loader = eval_loader,
        logger = train_logger,
        loss_fn = get_loss_fn(num_labels_protected),
        metrics = get_metrics(num_labels_protected),
        num_epochs = args_attack.num_epochs,
        lr = args_attack.learning_rate
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--raw", type=bool, default=False, help="")
    base_args = parser.parse_args()
    
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)    
    args_train = argparse.Namespace(**{**cfg["train_config_baseline"], **cfg["data_config"]})
    args_attack = argparse.Namespace(**cfg["adv_attack"])

    main(base_args, args_train, args_attack)


