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
from src.training_logger import TrainLogger
from src.utils import get_metrics, get_loss_fn, get_num_labels

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(base_args, args):    
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    num_labels = get_num_labels(args.labels_task_path)
    num_labels_protected = get_num_labels(args.labels_protected_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
        "baseline" if base_args.baseline else "diff_pruning",
        args.model_name.split('/')[-1],
        str(args.batch_size),
        str(args.learning_rate),
        str(args.adv_rev_ratio)
    ])
    train_logger = TrainLogger(
        log_dir = Path("logs")  / "adv_training",
        logger_name = logger_name,
        logging_step = args.logging_step
    )

    trainer_cls = AdverserialNetwork if base_args.baseline else DiffNetwork
    
    trainer = trainer_cls(
        args.model_name,
        num_labels,
        num_labels_protected,
        adv_count=args.adv_count,
        adv_rev_ratio=args.adv_rev_ratio
    )        
    trainer.to(DEVICE)
        
    if base_args.baseline:
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            get_loss_fn(num_labels_protected),
            get_metrics(num_labels_protected),
            args.num_epochs,
            args.learning_rate,
            args.learning_rate_adverserial,
            args.warmup_steps,
            args.max_grad_norm,
            args.output_dir,
            args.bitfit
        )
    else:    
        trainer.fit(
            train_loader,
            eval_loader,
            train_logger,
            get_loss_fn(num_labels),
            get_metrics(num_labels),
            get_loss_fn(num_labels_protected),
            get_metrics(num_labels_protected),
            args.num_epochs_finetune,
            args.num_epochs_fixmask,
            args.alpha_init,
            args.concrete_lower,
            args.concrete_upper,
            args.structured_diff_pruning,
            args.sparsity_pen,
            args.fixmask_pct,
            args.learning_rate,
            args.learning_rate_alpha,
            args.learning_rate_adverserial,
            args.optimizer_warmup_steps,
            args.weight_decay,
            args.adam_epsilon,
            args.max_grad_norm,
            args.output_dir
        )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--baseline", type=bool, default=False, help="")
    parser.add_argument("--raw", type=bool, default=False, help="")
    base_args = parser.parse_args()
    
    cfg_name = "train_config_adv_baseline" if base_args.baseline else "train_config"
    
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)    
    args = argparse.Namespace(**{**cfg[cfg_name], **cfg["data_config"]})

    main(base_args, args)


