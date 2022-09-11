from pathlib import Path
from functools import reduce
from copy import deepcopy
import torch
from transformers import AutoModel

from src.models.model_base import BaseModel, BasePruningModel
from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.models.model_modular import ModularModel

from typing import Optional, Union


def merge_models(model_list: list) -> torch.nn.Module:
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


def merge_diff_models(
    diff_model_list: list,
    base_model: Optional[torch.nn.Module] = None,
    only_first: bool = False
) -> BaseModel:
    model_name = diff_model_list[0].model_name
    model_list = []
    for m in diff_model_list:
        if only_first:
            idx_list = [0]
        else:
            idx_list = list(range(m.n_parametrizations))
        for idx in idx_list:
            model = m.get_diff_weights(idx, as_module=True)
            model = BaseModel(model_name, model.state_dict())
            model_list.append(model)
    sd = base_model.encoder.state_dict() if base_model is not None else AutoModel.from_pretrained(model_name)
    model_list.append(BaseModel(model_name, sd))
    return merge_models(model_list)


def load_cp(
    cp_path: Union[str, Path],
    cp_is_sd: bool,
    cp_model_type: Union[str, BaseModel],
    cp_modular_biased: Optional[bool] = None
):
    if cp_path is not None:
        if cp_is_sd:
            return torch.load(cp_path)
        assert cp_model_type is not None, "if cp_path is set cp_model_type needs to be set as well"
        if isinstance(cp_model_type, str):
            model_class = eval(cp_model_type)
        else:
            model_class = cp_model_type
        kwargs = {"filepath": cp_path}
        if isinstance(model_class, BasePruningModel):
            kwargs["remove_parametrizations"] = True
            if isinstance(model_class, ModularDiffModel):
                assert cp_modular_biased is not None, "if model type is Modular, cp_modular_debiased needs to be set"
                kwargs["debiased"] = not cp_modular_biased
        m = model_class.load_checkpoint(**kwargs)
        return m.encoder.state_dict()