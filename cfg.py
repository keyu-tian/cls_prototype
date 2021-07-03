import yaml
from easydict import EasyDict


def parse_cfg(cfg_path, world_size, rank, only_model=False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    if only_model:
        return cfg.model
    
    cp, ls_k, ls = None, None, None
    for comp_k, comp in cfg.items():
        if comp_k != 'data' and isinstance(comp, dict):
            for k, v in comp.items():
                if isinstance(v, list):
                    assert len(v) == world_size
                    cp, ls_k, ls = comp, k, v
    
    if cp is not None:
        cp[ls_k] = ls[rank]
        cfg.train.descs = [f'[rk{rk:02d}: {ls_k}={ls[rk]}]' for rk in range(world_size)]
        cfg.train.descs_key = ls_k
        cfg.train.descs_val = cp[ls_k]
    else:
        cfg.train.descs = None
    
    return cfg
