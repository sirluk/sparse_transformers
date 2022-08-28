import ruamel.yaml as yaml
import argparse
import torch

from main import main

torch.manual_seed(0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--baseline", type=bool, default=False, help="")
    parser.add_argument("--raw", type=bool, default=False, help="")
    base_args = parser.parse_args()
    
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg_name = "train_config_adv_baseline" if base_args.baseline else "train_config"
    
    adv_rev_ratios = [0,1,2,3,5,8,13]
    
    for adv_rev_ratio in adv_rev_ratios:
        cfg[cfg_name]["adv_rev_ratio"] = adv_rev_ratio
        args_train = argparse.Namespace(**{**cfg[cfg_name], **cfg["data_config"]})
        args_attack = argparse.Namespace(**cfg["adv_attack"])

        main(base_args, args_train, args_attack)
