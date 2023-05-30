import os
from typing import Dict, Any

from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning.core.saving import load_hparams_from_yaml


def load_hparams(hp: str, is_root: bool = True) -> DictConfig:
    if hp is None:
        raise ValueError(
            "`hp` must not be None."
        )
    hp = os.path.abspath(hp)
    if not os.path.exists(hp):
        raise ValueError(
            f"`hp` ({hp}) does not exist."
        )
    if os.path.isdir(hp):
        raise ValueError(
            f"`hp` ({hp}) should be a file instead of a directory."
        )
    hp = load_hparams_from_yaml(hp)
    if is_root:
        for key in ("trainer", "model", "data"):
            if isinstance(getattr(hp, key), str):
                try:
                    setattr(hp, key, load_hparams(getattr(hp, key), is_root=False))
                except ConfigAttributeError:
                    raise ValueError(
                        f"`{key}` should be included in the hparams file."
                    )
    return hp


if __name__ == "__main__":
    from pprint import pprint
    pprint(        
        load_hparams("/Users/saner/Desktop/第四节课/pl_mltc/hparams/lstm.yaml")
    )
