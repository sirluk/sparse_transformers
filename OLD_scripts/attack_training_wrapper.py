import ruamel.yaml as yaml
import argparse
import torch

from attack_training import main

torch.manual_seed(0)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="Whether to run on small subset for testing")
    parser.add_argument("--raw", type=bool, default=False, help="")
    base_args = parser.parse_args()
    
    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    
    checkpoints = [
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_1.0-baseline.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_2.0-baseline.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_3.0-baseline.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_5.0-baseline.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_10.0-baseline.pt",
        "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_30.0-baseline.pt",
        "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_100.0-baseline.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_1.0-diff_pruning.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_2.0-diff_pruning.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_3.0-diff_pruning.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_5.0-diff_pruning.pt",
        # "bert_uncased_L-2_H-128_A-2-adv_rev_ratio_10.0-diff_pruning.pt"
    ]
    
    for checkpoint in checkpoints:
        print(f"adv attack for {checkpoint}")
        cfg["adv_attack"]["checkpoint_path"] = "checkpoints/" + checkpoint
        args = argparse.Namespace(**{**cfg["adv_attack"], **cfg["data_config"]})

        main(base_args, args)
