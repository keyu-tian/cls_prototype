import argparse
import os
import time
from datetime import datetime
from pprint import pformat

import numpy as np
import colorama
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchsummary import torchsummary

from cfg import parse_cfg
from data import Scene15Set
from dist import TorchDistManager
from ema import EMA
from log import create_loggers
from meta import NUM_CLASSES
from models import build_model
from utils import filter_params, TopKHeap, adjust_learning_rate, AverageMeter, master_echo
from loss import LabelSmoothCELoss, mixup_data, mixup_criterion


def main():
    colorama.init(autoreset=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    parser = argparse.ArgumentParser(description='IPCV homework set 3 -- scene cls task')
    parser.add_argument('--main_py_rel_path', type=str, required=True)
    parser.add_argument('--exp_dirname', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    args = parser.parse_args()
    
    sh_root = os.getcwd()
    exp_root = os.path.join(sh_root, args.exp_dirname)
    os.chdir(args.main_py_rel_path)
    prj_root = os.getcwd()
    os.chdir(sh_root)
    
    dist = TorchDistManager(args.exp_dirname, 'auto', 'auto')
    loggers = create_loggers(prj_root, sh_root, exp_root, dist)
    
    cfg = parse_cfg(args.cfg, dist.world_size, dist.rank)
    if dist.is_master():
        try:
            main_process(exp_root, cfg, dist, loggers)
        except Exception as e:
            loggers[1].log(pr=-1., rem=0)
            raise e
    else:
        try:
            main_process(exp_root, cfg, dist, loggers)
        except Exception:
            exit(-1)


def main_process(exp_root, cfg, dist, loggers):
    data_cfg, model_cfg, train_cfg = cfg.data, cfg.model, cfg.train
    
    loggers[1].log(
        net=model_cfg.name, bmo=model_cfg.kwargs.bn_mom, dr=model_cfg.kwargs.dropout_rate,
        bs=data_cfg.batch_size, rot=data_cfg.rot, r=data_cfg.scale_ratio,
        ep=train_cfg.epochs, lr=train_cfg.lr, wd=train_cfg.wd, nowd=train_cfg.nowd, ls=train_cfg.ls_ratio, mix=train_cfg.get('mix', 0), clp=train_cfg.grad_clip,
        pr=0, rem=0, beg_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    get_new_tr_loader, te_loader = build_dataloader(data_cfg)
    ema, model = build_model(model_cfg)
    if dist.is_master():
        torchsummary.summary(model, (3, 224, 224))

    loggers[0].info(f'=> [final cfg]:\n{pformat(dict(cfg))}')
    train_model(exp_root, train_cfg, dist, loggers, get_new_tr_loader, te_loader, ema, model)


def build_dataloader(data_cfg):
    if data_cfg.root is None:
        data_cfg.root = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', 'scene15'))
    te_set = Scene15Set(root_dir_path=data_cfg.root, train=False, vgg=data_cfg.vgg, val_crop=data_cfg.val_crop)
    te_loader = DataLoader(te_set, data_cfg.batch_size * 4, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    ranges = dict(
        wh=np.linspace(data_cfg.wh[0], data_cfg.wh[1], 10),
        scale_ratio=np.linspace(data_cfg.scale_ratio[0], data_cfg.scale_ratio[1], 10),
        sharp=np.linspace(data_cfg.sharp[0], data_cfg.sharp[1], 10),
        trans=np.linspace(data_cfg.trans[0], data_cfg.trans[1], 10),
        rot=np.linspace(data_cfg.rot[0], data_cfg.rot[1], 10),
        jitter=np.linspace(data_cfg.jitter[0], data_cfg.jitter[1], 10),
    )
    
    def get_new_tr_loader(idx10, tb_lg):
        kw = {k: round(v[idx10].item(), 3) for k, v in ranges.items()}
        tr_set = Scene15Set(
            root_dir_path=data_cfg.root, train=True, vgg=data_cfg.vgg,
            **kw
        )
        augmented = torch.stack([tr_set[7][0] for _ in range(6)])
        if tb_lg is not None:
            tb_lg.add_images('augmented', tr_set.denormalize(augmented.data).cpu().numpy(), idx10, dataformats='NCHW')
        return kw, DataLoader(tr_set, data_cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    return get_new_tr_loader, te_loader


@torch.no_grad()
def eval_model(te_loader, model: torch.nn.Module, cuda=True):
    tr = model.training
    model.train(False)
    tot_correct, tot_pred, tot_loss, tot_iters = 0, 0, 0., 0
    if cuda:
        bar = te_loader
    else:
        from tqdm import tqdm
        bar = tqdm(te_loader)
    for (inp, tar) in bar:
        if cuda:
            inp, tar = inp.cuda(non_blocking=True), tar.cuda(non_blocking=True)
        logits = model(inp)
        tot_pred += tar.shape[0]
        tot_correct += logits.argmax(dim=1).eq(tar).sum().item()
        tot_loss += F.cross_entropy(logits, tar).item()
        tot_iters += 1
        if not cuda:
            bar.set_postfix(dict(cur_acc=100. * tot_correct / tot_pred))
    model.train(tr)
    return 100. * tot_correct / tot_pred, tot_loss / tot_iters


def train_model(exp_root, train_cfg, dist, loggers, get_new_tr_loader, te_loader, ema: EMA, model):
    # todo: mix-up
    lg, st_lg, tb_lg = loggers
    try:
        [tb_lg.add_scalar(f'dist/{train_cfg.descs_key}', float(train_cfg.descs_val), t) for t in [0, 1]]
    except ValueError:
        pass
    
    prefix = train_cfg.descs[dist.rank] if train_cfg.descs is not None else f'rk{dist.rank}'
    prefix = prefix.replace(' ', '').replace(':', '_').replace('=', '').strip('[]')
    saved_path = os.path.join(exp_root, f'{prefix}_best_ckpt.pth')
    all_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = filter_params(model) if train_cfg.nowd else all_params
    optimizer = torch.optim.SGD(params, lr=float(train_cfg.lr), weight_decay=float(train_cfg.wd), momentum=0.9, nesterov=True)
    loss_fn = LabelSmoothCELoss(float(train_cfg.ls_ratio), NUM_CLASSES) if train_cfg.ls_ratio is not None else CrossEntropyLoss()
    loss_fn = loss_fn.cuda()
    
    aug_kw, tr_loader = get_new_tr_loader(0, tb_lg if dist.is_master() else None)
    tr_iters = len(tr_loader)
    max_ep = train_cfg.epochs
    max_iter = max_ep * tr_iters
    saved_acc, best_acc, best_acc_ema = -1, -1, -1
    topk_accs, topk_accs_ema = TopKHeap(maxsize=30), TopKHeap(maxsize=30)
    epoch_speed = AverageMeter(4)
    
    mix = train_cfg.mix if 'mix' in train_cfg and 0 < train_cfg.mix < 1 else None
    loop_start_t = time.time()
    milestone_ep: list = np.linspace(0, max_ep, 10+1, dtype=int)[1:-1].tolist()
    for ep in range(max_ep):
        te_freq = max(1, round(tr_iters // 3)) if ep >= max_ep * 0.5 else tr_iters * 4
        ep_str = f'%{len(str(max_ep))}d'
        ep_str %= ep + 1
        ep_str = f'ep[{ep_str}/{max_ep}]'
        if ep == max_ep - 1:
            master_echo(dist.is_master(), f' @@@@@ {exp_root}      be={best_acc:5.2f} ({best_acc_ema:5.2f})', '36')
        
        # train one epoch
        if ep in milestone_ep:  # update tr_loader (replace the current augmentation policy to a stronger one)
            torch.cuda.empty_cache()
            aug_kw, tr_loader = get_new_tr_loader(milestone_ep.index(ep) + 1, tb_lg if dist.is_master() else None)
            master_echo(dist.is_master(), f' @@@@@ {exp_root}, akw={aug_kw}     be={best_acc:5.2f} ({best_acc_ema:5.2f})', '36')
            
        ep_start_t = time.time()
        last_t = time.time()
        for it, (inp, tar) in enumerate(tr_loader):
            it_str = f'%{len(str(tr_iters))}d'
            it_str %= it + 1
            it_str = f'it[{it_str}/{tr_iters}]'
            cur_iter = it + ep * tr_iters
            data_t = time.time()
            
            inp, tar = inp.cuda(non_blocking=True), tar.cuda(non_blocking=True)
            cuda_t = time.time()
            
            if mix is not None:
                inp, tar_a, tar_b, lam = mixup_data(inp, tar, train_cfg.mix)
                logits = model(inp)
                loss = mixup_criterion(loss_fn, logits, tar_a, tar_b, lam)
            else:
                logits = model(inp)
                loss = loss_fn(logits, tar)
            forw_t = time.time()
            
            loss.backward()
            back_t = time.time()
            
            orig_norm = float(torch.nn.utils.clip_grad_norm_(all_params, float(train_cfg.grad_clip)))
            clip_t = time.time()
            
            sche_lr = adjust_learning_rate(optimizer, cur_iter, max_iter, train_cfg.lr)
            actual_lr = sche_lr * min(1., float(train_cfg.grad_clip) / orig_norm)
            optimizer.step()
            optimizer.zero_grad()
            optm_t = time.time()
            
            ema.step(model, cur_iter + 1)
            
            logging = cur_iter == max_iter - 1 or cur_iter % te_freq == 0
            if logging or orig_norm > 15 or cur_iter < tr_iters:
                tb_lg.add_scalars('opt/lr', {'sche': sche_lr, 'actu': actual_lr}, cur_iter)
                tb_lg.add_scalars('opt/norm', {'orig': orig_norm, 'clip': train_cfg.grad_clip}, cur_iter)
            
            if logging:
                preds = logits.detach().argmax(dim=1)
                train_acc = 100. * preds.eq(tar).sum().item() / tar.shape[0]
                train_loss = loss.item()
                
                test_acc, test_loss = eval_model(te_loader, model)
                topk_accs.push_q(test_acc)
                best_acc = max(best_acc, test_acc)
                
                ema.load_ema(model)
                test_acc_ema, test_loss_ema = eval_model(te_loader, model)
                best_acc_ema = max(best_acc_ema, test_acc_ema)
                if best_acc_ema > saved_acc:
                    saved_acc = best_acc_ema;
                    torch.save(model.state_dict(), saved_path)
                ema.recover(model)
                topk_accs_ema.push_q(test_acc_ema)
                
                test_t = time.time()
                
                if best_acc > saved_acc:
                    saved_acc = best_acc;
                    torch.save(model.state_dict(), saved_path)
                
                remain_time, finish_time = epoch_speed.time_preds(max_ep - (ep + 1))
                
                lg.info(
                    f'=> {ep_str}, {it_str}:    lr={sche_lr:.3g}({actual_lr:.3g}), nm={orig_norm:.1f}\n'
                    f'  [train] L={train_loss:.3f}, acc={train_acc:5.2f}, da={data_t - last_t:.3f} cu={cuda_t - data_t:.3f} fp={forw_t - cuda_t:.3f} bp={back_t - forw_t:.3f} cl={clip_t - back_t:.3f} op={optm_t - clip_t:.3f} te={test_t-optm_t:.3f}\n'
                    f'  [test ] L={test_loss:.3f}({test_loss_ema:.3f}), acc={test_acc:5.2f}({test_acc_ema:5.2f})      remain [{str(remain_time)}] ({finish_time})       >>> [best]={best_acc:5.2f}({best_acc_ema:5.2f})'
                )
                tb_lg.add_scalar('test/acc', test_acc, cur_iter)
                tb_lg.add_scalars('test/acc', {'ema': test_acc_ema}, cur_iter)
                tb_lg.add_scalar('test/loss', test_loss, cur_iter)
                tb_lg.add_scalars('test/loss', {'ema': test_loss_ema}, cur_iter)
                tb_lg.add_scalar('train/acc', train_acc, cur_iter)
                tb_lg.add_scalar('train/loss', train_loss, cur_iter)
                
                st_lg.log(
                    pr=(cur_iter + 1) / max_iter,
                    clr=sche_lr, nm=orig_norm,
                    tr_L=train_loss, te_L=test_loss, em_L=test_loss_ema,
                    tr_A=train_acc, te_A=test_acc, em_A=test_acc_ema,
                    be=best_acc, be_e=best_acc_ema,
                    rem=remain_time.seconds, end_t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_time.seconds)),
                )
            
            last_t = time.time()
        # iteration loop end
        epoch_speed.update(time.time() - ep_start_t)
        tb_lg.add_scalar('test_ep_best/acc', best_acc, ep + 1)
        tb_lg.add_scalars('test_ep_best/acc', {'ema': best_acc_ema}, ep + 1)
    
    # epoch loop end
    
    topk_test_acc = sum(topk_accs) / len(topk_accs)
    topk_test_acc_ema = sum(topk_accs_ema) / len(topk_accs_ema)
    
    topk_accs = dist.dist_fmt_vals(topk_test_acc, None)
    topk_accs_ema = dist.dist_fmt_vals(topk_test_acc_ema, None)
    best_accs = dist.dist_fmt_vals(best_acc, None)
    best_accs_ema = dist.dist_fmt_vals(best_acc_ema, None)
    if dist.is_master():
        [tb_lg.add_scalar('z_final_best/topk_accs', topk_accs.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_best/topk_accs_ema', topk_accs_ema.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_best/best_accs', best_accs.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_best/best_accs_ema', best_accs_ema.max().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/topk_accs', topk_accs.mean().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/topk_accs_ema', topk_accs_ema.mean().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/best_accs', best_accs.mean().item(), e) for e in [-max_ep, max_ep]]
        [tb_lg.add_scalar('z_final_mean/best_accs_ema', best_accs_ema.mean().item(), e) for e in [-max_ep, max_ep]]
    
    if train_cfg.descs is not None:
        perform_dict_str = pformat({
            des: f'topk={ta.item():5.2f}({tae.item():5.2f}), best={ba.item():5.2f}({bae.item():5.2f})'
            for des, ta, tae, ba, bae in zip(train_cfg.descs, topk_accs, topk_accs_ema, best_accs, best_accs_ema)
        })
        perform_dict_str = f'{perform_dict_str}\n\n'
    else:
        perform_dict_str = ''
    
    eval_str = (
        f' mean-top     @ (max={topk_accs.max():5.2f}, mean={topk_accs.mean():5.2f}, std={topk_accs.std():.2g}) {str(topk_accs).replace(chr(10), " ")})\n'
        f' EMA mean-top @ (max={topk_accs_ema.max():5.2f}, mean={topk_accs_ema.mean():5.2f}, std={topk_accs_ema.std():.2g}) {str(topk_accs_ema).replace(chr(10), " ")})\n'
        f' best         @ (max={best_accs.max():5.2f}, mean={best_accs.mean():5.2f}, std={best_accs.std():.2g}) {str(best_accs).replace(chr(10), " ")})\n'
        f' EMA best     @ (max={best_accs_ema.max():5.2f}, mean={best_accs_ema.mean():5.2f}, std={best_accs_ema.std():.2g}) {str(best_accs_ema).replace(chr(10), " ")})'
    )
    
    dt = time.time() - loop_start_t
    lg.info(
        f'=> training finished,'
        f' total time cost: {dt / 60:.2f}min ({dt / 60 / 60:.2f}h)\n'
        f' performance: \n{perform_dict_str}{eval_str}'
    )
    
    st_lg.log(
        pr=1., rem=0,
        m_tk=topk_accs.mean().item(), m_tk_e=topk_accs_ema.mean().item(),
        m_be=best_accs.mean().item(), m_be_e=best_accs_ema.mean().item(),
        end_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )
    
    dist.barrier()
    tb_lg.close()


if __name__ == '__main__':
    import numpy as np
    from torch import multiprocessing as mp
    main()
