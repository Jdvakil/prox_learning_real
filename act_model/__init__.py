"""Local, checkpoint-compatible ACT model; no prox_learning checkout is required."""
from argparse import Namespace


def build_model(config):
    from .detr_vae import build

    defaults = dict(backbone="resnet18", position_embedding="sine", dilation=False,
                    masks=False, lr_backbone=1e-5, dropout=0.1, pre_norm=False,
                    n_proximity_sensors=0, prox_tokens_per_sensor=1, prox_feat_dim=3)
    defaults.update(config)
    defaults["lr_backbone"] = float(defaults["lr_backbone"])
    return build(Namespace(**defaults))
