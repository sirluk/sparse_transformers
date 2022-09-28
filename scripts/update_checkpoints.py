import sys
sys.path.insert(0,'..')

import os
import re
import torch
from tqdm import tqdm

from src.models.model_diff_modular import ModularDiffModel
from src.models.model_diff_adv import AdvDiffModel
from src.models.model_diff_task import TaskDiffModel
from src.models.model_adv import AdvModel
from src.models.model_task import TaskModel
from src.models.model_modular import ModularModel


PATH = "/share/home/lukash/pan16/bertl4/cp"
CLS_MAP = {
    TaskModel: re.compile(r"task-baseline"),
    AdvModel: re.compile(r"adverserial-baseline"),
    TaskDiffModel: re.compile(r"task-diff_pruning"),
    AdvDiffModel: re.compile(r"adverserial-diff_pruning"),
    ModularModel: re.compile(r"modular-.*-baseline"),
    ModularDiffModel: re.compile(r"modular-.*-diff_pruning")
}

experiments = os.listdir(PATH)
main_iter = tqdm(CLS_MAP.items(), leave=False, position=0)
for m_cls, pat in main_iter:
    main_iter.set_description(m_cls.__name__)
    for exp_name in tqdm([e for e in experiments if re.match(pat, e)], leave=False, position=1):
        filepath = os.path.join(PATH, exp_name)
        info_dict = torch.load(filepath, map_location="cpu")
        if "sparsity_pen" not in info_dict:
            info_dict["sparsity_pen"] = 1.25e-7
            torch.save(info_dict, filepath)
        # import IPython; IPython.embed(); exit(1)
        