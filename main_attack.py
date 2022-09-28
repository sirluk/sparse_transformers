import ruamel.yaml as yaml
import argparse
from pathlib import Path
import torch

from src.training_logger import TrainLogger
from src.adv_attack import run_adv_attack
from src.data_handler import get_data
from src.model_functions import model_factory
from src.utils import (
    get_device,
    set_num_epochs_debug,
    set_dir_debug,
    set_optional_args
)

torch.manual_seed(0)


### DEFINE MANUALLY
CP_DIR = "/share/home/lukash/pan16/bertl4/cp/"
# CP = "task-baseline-bert_uncased_L-4_H-256_A-4-64-2e-05-seed4.pt"
CP = "task-diff_pruning_0.05-bert_uncased_L-4_H-256_A-4-64-2e-05-seed4.pt"
LOAD_CP_KWARGS = {"remove_parametrizations": True}
### DEFINE MANUALLY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Whether to run on small subset for testing")
    parser.add_argument("--gpu_id", nargs="*", type=int, default=[0], help="")
    parser.add_argument("--seed", type=int, default=0, help="torch random seed")
    parser.add_argument("--ds", type=str, default="bios", help="dataset")
    parser.add_argument("--cpu", action="store_true", help="Run on cpu")
    parser.add_argument("--no_weighted_loss", action="store_true", help="do not use weighted loss for protected attribute")
    base_args, optional = parser.parse_known_args()

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{base_args.ds}"
    args_train = argparse.Namespace(**cfg["adv_attack"], **cfg[data_cfg], **cfg["model_config"])

    set_optional_args(args_train, optional)

    if base_args.debug:
        set_num_epochs_debug(args_train)
        set_dir_debug(args_train)

    torch.manual_seed(base_args.seed)
    print(f"torch.manual_seed({base_args.seed})")

    device = get_device(not base_args.cpu, base_args.gpu_id)
    print(f"Device: {device}")

    train_loader, val_loader, _, num_labels_protected, protected_key, protected_class_weights = get_data(
        args_train = args_train,
        use_all_attr = True,
        compute_class_weights = (not base_args.no_weighted_loss),
        device = device[0],
        debug = base_args.debug
    )

    trainer = model_factory(f"{CP_DIR}/{CP}", **LOAD_CP_KWARGS)
    trainer.to(device)

    logger_name = "-".join([x for x in [
        f"only_adv_attack_{CP[:-3]}",
        str(args_train.batch_size),
        str(args_train.learning_rate),
        "weighted_loss_prot" if not base_args.no_weighted_loss else None,
        f"seed{base_args.seed}"
    ] if x is not None])
    train_logger = TrainLogger(
        log_dir = Path(args_train.log_dir),
        logger_name = logger_name,
        logging_step = args_train.logging_step
    )

    print(f"running model {CP}")

    run_adv_attack(
        base_args = base_args,
        args_train = args_train,
        args_attack = args_train,
        trainer = trainer,
        train_logger = train_logger,
        train_loader = train_loader,
        val_loader = val_loader,
        num_labels_protected = num_labels_protected,
        protected_key = protected_key,
        protected_class_weights = protected_class_weights
    )
    

if __name__ == "__main__":

    main()


