import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from cfg import parse_cfg
from data import Scene15Set
from main import eval_model
from models import build_model


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    parser = argparse.ArgumentParser(description='IPCV homework set 3 -- scene cls task')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()

    model_cfg = parse_cfg(args.cfg, None, None, only_model=True)
    ema, model = build_model(model_cfg)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    
    for val_crop in (True, False):
        te_set = Scene15Set(root_dir_path='/mnt/lustre/tiankeyu/datasets/scene15', train=False, vgg='vgg' in model_cfg.name, val_crop=val_crop)
        te_loader = DataLoader(te_set, 64, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        test_acc, test_loss = eval_model(te_loader, model)
        print(f'[val_crop={val_crop}] test_acc={test_acc:5.2f}, test_loss={test_loss:.2f}')


if __name__ == '__main__':
    main()
