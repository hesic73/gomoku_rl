import wandb

from omegaconf import DictConfig, OmegaConf
import datetime


def dict_flatten(a: dict, delim: str = "."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg: DictConfig):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg: DictConfig = cfg.wandb
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        mode=wandb_cfg.get("mode", "disabled"),
    )
    kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict)
    return run
