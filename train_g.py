import argparse
import expman
import h5py
import numpy as np
import pandas as pd
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, ReduceLROnPlateau
from tqdm import trange

from model_g import get_model

# paths to HDF5 file containing features
DATA_PATHS = {
    'yfcc100m-hybridfc6': 'data/yfcc100m-hybridfc6/features-001-of-100.h5'
}

@torch.no_grad()
def prepare(features, args):

    features, start, end = features
    n_features = end - start
    
    n = args.dim  # number of pivots
    b = args.batch_size
    d = features.shape[1]  # dimensionality of original features

    # takes n features as pivots and two batches of features as data points
    n_take = n + 2*b
    if False:  # random access to hdf5 is slow
        x = np.random.choice(n_features, n_take, replace=False) + start
        
        # sort indexes for h5py, keep also inverse sorting to return to random order
        s = np.argsort(x)
        invs = np.argsort(s)

        # get the features
        x = features[x[s], ...][invs]
    else:
        x = np.random.randint(n_features - n_take)
        x = features[x:x + n_take, ...]
        np.random.shuffle(x)
        
    x = torch.from_numpy(x).to(args.device)
    o1, o2, pivots = x.split((b, b, n))

    # object-object euclidean distances (target)
    oo = torch.pow(o1 - o2, 2).sum(1, keepdim=True).sqrt()

    # object-pivot euclidean distances (input)
    op1 = torch.cdist(o1, pivots, p=2)
    op2 = torch.cdist(o2, pivots, p=2)
    
    # pivot-pivot distances (additional input)
    pp = torch.pdist(pivots, p=2) 

    return op1, op2, pp, oo


def train(features, model, optimizer, scheduler, args):
    model.train()
    optimizer.zero_grad()

    real = []
    estimates = []
    
    steps_for_update = (args.accumulate // args.batch_size)
    steps = steps_for_update * args.iterations
    progress = trange(steps, disable=args.no_progress)
    for it in progress:
        op1, op2, pp, oo = prepare(features, args)  # already moved to device
        
        emb1, emb2 = model(op1, pp), model(op2, pp)
        oo_p = torch.pow(emb1 - emb2, 2).sum(1, keepdim=True).sqrt()

        mse = F.mse_loss(oo_p, oo)
        mae = F.l1_loss(oo_p, oo)
        mape = ((oo_p - oo) / (oo + 1e-8)).abs().mean()
        sml1 = F.smooth_l1_loss(oo_p, oo)
        
        progress.set_postfix({
            'mse': f'{mse.item():3.2f}',
            'mae': f'{mae.item():3.2f}',
            'mape': f'{mape.item():3.2f}',
            'sml1': f'{sml1.item():3.2f}'
        })

        if args.loss == 'mse':
            mse.backward()
        elif args.loss == 'mae':
            mae.backward()
        elif args.loss == 'mape':
            mape.backward()
        elif args.loss == 'sml1':
            sml1.backward()

        if (it + 1) % steps_for_update:
            optimizer.step()
            optimizer.zero_grad()
            if args.lr_schedule == 'cycle':
                scheduler.step()

        real.append(oo)
        estimates.append(oo_p.detach())
    
    real = torch.cat(real, 0).squeeze()
    estimates = torch.cat(estimates, 0).squeeze()

    return compute_metrics(real, estimates, prefix='train_')


@torch.no_grad()
def evaluate(features, model, args):
    model.eval()

    real = []
    estimates = []

    steps = (args.accumulate // args.batch_size) * args.val_iterations
    for _ in trange(steps, disable=args.no_progress):
        op1, op2, pp, oo = prepare(features, args)
        
        emb1, emb2 = model(op1, pp), model(op2, pp)
        oo_p = torch.pow(emb1 - emb2, 2).sum(1, keepdim=True).sqrt()

        real.append(oo)
        estimates.append(oo_p)
    
    real = torch.cat(real, 0).squeeze()
    estimates = torch.cat(estimates, 0).squeeze()
    
    return compute_metrics(real, estimates)


@torch.no_grad()
def compute_metrics(real, estimates, prefix=''):
    errors = estimates - real
    
    abs_errors = errors.abs()
    rel_errors = (errors / (real + 1e-8)).abs()
    sq_errors = torch.pow(errors, 2)
    sm_abs_errors = F.smooth_l1_loss(estimates, real, reduction='none')

    div = real / estimates

    return {
        prefix + 'mse': sq_errors.mean().item(),
        prefix + 'mse_std': sq_errors.std().item(),
        
        prefix + 'mape': rel_errors.mean().item(),
        prefix + 'mape_std': rel_errors.std().item(),
        
        prefix + 'mae': abs_errors.mean().item(),
        prefix + 'mae_std': abs_errors.std().item(),

        prefix + 'sml1': sm_abs_errors.mean().item(),
        prefix + 'sml1_std': sm_abs_errors.std().item(),

        prefix + 'dist': (div.max() / div.min()).item(),
        prefix + 'mdist': (div / div.min()).mean().item(),
        prefix + 'mdist_std': (div / div.min()).std().item()
    }


def main(args):

    exp = expman.Experiment(args, root=args.rundir, ignore=('cuda', 'device', 'epochs', 'no_progress', 'resume', 'rundir'))
    print(exp)

    ckpt_dir = exp.path_to('ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    features = h5py.File(DATA_PATHS[args.data], 'r')['data']
    
    # a subset is (set, start, end)
    train_features = (features, 0, 750 * (10 ** 3))
    val_features = (features, 750 * (10 ** 3), 900 * (10 ** 3))
    # test_features = (features, 900 * (10 ** 3), 1000 * (10 ** 3))

    # Build the model
    model = get_model(args).to(args.device)
    
    # Optimizer
    if args.optim == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    
    # LR Scheduler
    if args.lr_schedule == 'fixed':
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)  # no-op scheduler
    elif args.lr_schedule == 'cycle':
        scheduler = CyclicLR(optimizer, base_lr=(args.lr / 100), max_lr=args.lr, mode='triangular2')
    elif args.lr_schedule == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=args.patience)

    start_epoch = 0
    # resume from checkpoint?
    if args.resume:
        ckpt = exp.ckpt('last.pth')
        if not os.path.exists(ckpt):
            print("No checkpoint to resume from: starting from scratch...")
        else:
            ckpt = torch.load(ckpt)
            start_epoch = ckpt['metrics']['epoch']
            print('Resuming from epoch:', start_epoch)
            
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

    # Train loop
    progress = trange(start_epoch + 1, args.epochs + 1, initial=start_epoch, total=args.epochs, disable=args.no_progress)
    for epoch in progress:
        progress.set_description('TRAIN')
        train_metrics = train(train_features, model, optimizer, scheduler, args)

        progress.set_description('EVAL')
        eval_metrics = evaluate(val_features, model, args)
        
        metrics = {'epoch': epoch, **train_metrics, **eval_metrics}

        ckpt = exp.ckpt('last.pth')
        torch.save({
            'metrics': metrics,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, ckpt)

        if is_new_best(exp.log, metrics):
            best_ckpt = exp.ckpt(f'ckpt_e{epoch}.pth')
            shutil.copyfile(ckpt, best_ckpt)
            
        exp.push_log(metrics)

        if args.lr_schedule == 'plateau':
            scheduler.step(metrics['mse'])


def is_new_best(log, metrics):
    if log.empty:
        return True
    
    # any improvement on any of these metrics will trigger model snapshotting
    watch_metrics = ['mae', 'mape', 'mse', 'sml1']
    
    best = log[watch_metrics].min()
    current = pd.Series(metrics)[watch_metrics]
    return (current < best).any()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distance Estimation from Pivot Distances',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data params
    parser.add_argument('data', choices=('yfcc100m-hybridfc6',), help='dataset for training')
    parser.add_argument('-s', '--seed', type=int, default=23, help='Random initial seed')
    
    # Model params
    parser.add_argument('model', choices=('mlp', 'res-mlp'), help='dataset for training')
    parser.add_argument('-d', '--dim', type=int, default=200, help='Final dimensionality (also number of pivots)')
    parser.add_argument('-l', '--depth', type=int, default=2, help='Number of hidden layers (for MLP) or residual blocks (for ResMLP)')
    parser.add_argument('-f', '--fusion', choices=('early', 'mid', 'late'), default='early', help='When to fuse obect-pivot and pivot-pivot branches')
    parser.add_argument('-n', '--batch-norm', action='store_true', default=False, help='Whether to use BN in the model')
    parser.add_argument('-o', '--dropout', type=float, default=0.5, help='Dropout probability for hidden layers')
    parser.add_argument('--loss', choices=('mse', 'mape', 'mae', 'sml1'), default='mse', help='The metric to optimize')
    
    # Optimization params
    parser.add_argument('--optim', choices=('sgd', 'adam'), default='sgd', help='Optimizer to use')
    parser.add_argument('-b', '--batch-size', type=int, default=50, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='Number of iterations (optimizer steps) per epoch')
    parser.add_argument('-v', '--val-iterations', type=int, default=20, help='Number of iterations (each of size defined by -a) for validation')
    parser.add_argument('-a', '--accumulate', type=int, default=100, help='How many samples to accumulate before optimizer step (must be a multiple of batch size)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lr-schedule', choices=('fixed', 'cycle', 'plateau'), default='cycle', help='LR scheduling')
    parser.add_argument('--patience', type=int, default=10, help='Patience in epoch for "plateau" schedule')

    # Other
    parser.add_argument('-r', '--rundir', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('--resume', action='store_true', dest='resume', help='Resume training')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Run without CUDA')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')

    parser.set_defaults(cuda=True, no_progress=False, resume=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    main(args)
