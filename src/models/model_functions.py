from functools import reduce
from copy import deepcopy
import torch
from transformers import AutoModel


def merge_models(model_list: list):
    # assert all weights match
    sets = [set([n for n, _ in m.named_parameters()]) for m in model_list]
    try:
        intersect = sets[0].intersection(*sets[1:])
        assert len(sets[0]) == len(intersect)
    except:
        all_keys = sets[0].union(*sets[1:])
        missing = [k for k in all_keys if k not in intersect]
        raise Exception(f"Keys {missing} not present in all models")

    model_frame = deepcopy(model_list[0])
    with torch.no_grad():
        for p_name, p in model_frame.named_parameters():
            p.zero_()
            for i in range(len(model_list)):
                p_add = reduce(lambda a,b: getattr(a,b), [model_list[i]] + p_name.split("."))
                p += p_add

    return model_frame


def merge_diff_models(diff_model_list: list, only_first: bool):
    diff_weight_modules = []
    for m in diff_model_list:
        if only_first:
            idx_list = [0]
        else:
            idx_list = list(range(m.n_parametrizations))
        for idx in idx_list:
            diff_weight_modules.append(m.get_diff_weights(idx, as_module=True))
    model_list = [
        AutoModel.from_pretrained(diff_model_list[0].model_name),
        *diff_weight_modules
    ]
    return merge_models(model_list)