from pathlib import Path
from typing import Union, Optional, Dict, List
from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    delta: float = 1e-8

    @staticmethod
    def suffix_fn(suffix):
         return "" if len(suffix)==0 else f"_{suffix}"

    @staticmethod
    def is_best_check_fn(x, y, ascending, delta):
        return x < (y+delta) if ascending else x > (y-delta)

    def __init__(
        self,
        log_dir: Union[str, Path],
        logger_name: str,
        logging_step: int
    ):
        assert logging_step > 0, "logging_step needs to be > 0"

        self.logger_name = logger_name

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)

        self.writer = SummaryWriter(log_dir / logger_name)

        self.logging_step = logging_step

        self.reset()

    def validation_loss(self, eval_step: int, result: dict, suffix: str = ''):
        suffix = self.suffix_fn(suffix)
        for name, value in sorted(result.items(), key=lambda x: x[0]):
            self.writer.add_scalar(f'val/{name}{suffix}', value, eval_step)

    def step_loss(self, step: int, loss: Union[float, Dict[str, float]], lr: Optional[float] = None, increment_steps: bool = True, suffix: str = ''):
        self.steps += int(increment_steps)

        if not isinstance(loss, dict):
            loss = {suffix: loss}

        for k,v in loss.items():
            try:
                self.loss_dict[k] += v
            except KeyError:
                self.loss_dict[k] = v

        if (self.steps > 0) and (self.steps % self.logging_step == 0):
            logs = {f"step_loss{self.suffix_fn(k)}": v / self.steps for k,v in self.loss_dict.items()} 
            if lr:
                logs["step_learning_rate"] = lr
            for key, value in logs.items():
                self.writer.add_scalar(f'train/{key}', value, step // self.logging_step)            

            self.steps = 0
            self.loss_dict = {}

    def non_zero_params(self, step, n_p, n_p_zero, n_p_between, suffix: str = ''):
        suffix = self.suffix_fn(suffix)
        d = {
            "zero_ratio": n_p_zero / n_p,
            "between_ratio": n_p_between / n_p
        }
        for k,v in d.items():
            self.writer.add_scalar(f"train/{k}{suffix}", v, step)

    def log_best(
        self,
        result: dict,
        ascending: Union[bool, List[bool]],
        k: Optional[str] = None,
        suffix: str = ''
    ) -> None:
        if k is not None:
            result = {k: result[k]}
        
        if isinstance(ascending, bool):
            ascending = [ascending] * len(result)
        else:
            assert len(ascending)==len(result), f"len(result)={len(result)} but len(ascending)={len(ascending)}"

        suffix = self.suffix_fn(suffix)
        for k, v, asc in zip(result.keys(), result.values(), ascending):
            k = k + suffix
            try:
                if self.is_best_check_fn(v, self.best_eval_metric[k], asc, self.delta):
                    self.best_eval_metric[k] = v
            except:
                self.best_eval_metric[k] = v

    def is_best(
        self,
        result: dict,
        ascending: Union[bool, List[bool]],
        k: Optional[str] = None,
        weights: Optional[List] = None,
        binary: bool = False,
        suffix: str = '',
        log_best: bool = True
    ):
        if log_best: self.log_best(result, ascending, k, suffix)

        if k is not None:
            result = {k: result[k]}
            weights = [1]
        elif weights is None:
            weights = [1/len(result)] * len(result)
        else:
            assert len(result) == len(weights), "if passed weights needs to have same length as result"
        
        if isinstance(ascending, bool):
            ascending = [ascending] * len(result)
        else:
            assert len(ascending) == len(result), f"len(result)={len(result)} but len(ascending)={len(ascending)}"

        suffix = self.suffix_fn(suffix)
        res = 0.
        for name, current, asc, weight in zip(result.keys(), result.values(), ascending, weights):
            name = name + suffix
            if binary:
                res_ = self.is_best_check_fn(current, self.best_eval_metric[name], asc, self.delta) * 2 - 1
            else:
                best = (self.best_eval_metric[name] + self.delta)
                res_ = best - current if asc else current - best
            res += (res_ * weight)
        return res >= 0
            
    def write_best_eval_metric(self, k: Optional[str] = None):
        if k is None:
            d = self.best_eval_metric
        else:
            d = {k: self.best_eval_metric[k]}
        for key, v in d.items():
            self.writer.add_scalar(f'val/best_{key}', v)

    def reset(self):
        self.steps = 0
        self.loss_dict = {}
        self.best_eval_metric = {}