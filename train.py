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

from torch.optim import Adam
from tqdm import trange

from model import get_model

# paths to HDF5 file containing features
DATA_PATHS = {
    'yfcc100m-hybridfc6': 'data/yfcc100m-hybridfc6/features-001-of-100.h5'
}


@torch.no_grad()
def prepare(features, pivots, args):

    features, start, end = features
    n_features = end - start
    
    b = args.batch_size
    d = features.shape[1]
    n = pivots.shape[0]

    o1 = np.random.choice(n_features, b, replace=False) + start
    o2 = np.random.choice(n_features, b, replace=False) + start
    
    # sort indexes for h5py
    s1 = np.argsort(o1)
    s2 = np.argsort(o2)
    
    # keep also inverse sorting to return to random order
    invs1 = np.argsort(s1)
    invs2 = np.argsort(s2)

    o1 = torch.from_numpy(features[o1[s1], ...][invs1]).to(args.device)
    o2 = torch.from_numpy(features[o2[s2], ...][invs2]).to(args.device)

    # euclidean distances
    target = torch.pow(o1 - o2, 2).sum(1, keepdim=True)

    pivots = pivots.unsqueeze(0).expand(b, n, d)
    o1 = o1.unsqueeze(1).expand(b, n, d)
    o2 = o2.unsqueeze(1).expand(b, n, d)

    # pivot distances
    o1 = torch.pow(o1 - pivots, 2).sum(2)
    o2 = torch.pow(o2 - pivots, 2).sum(2)

    return o1, o2, target


def train(features, pivots, model, optimizer, args):
    model.train()
    optimizer.zero_grad()

    real = []
    estimates = []
    
    steps_for_update = (args.accumulate // args.batch_size)
    steps = steps_for_update * args.iterations
    progress = trange(steps)
    for it in progress:
        o1, o2, d = prepare(features, pivots, args)  # already moved to device        
        
        emb1, emb2 = model(o1), model(o2)
        dd = torch.pow(emb1 - emb2, 2).sum(1, keepdim=True)

        mse = F.mse_loss(dd, d)
        mape = ((dd - d) / (d + 1e-8)).abs().mean()
        
        progress.set_postfix({
            'mse': f'{mse.item():3.2f}',
            'mape': f'{mape.item():3.2f}'
        })

        mse.backward()

        if (it + 1) % steps_for_update:
            optimizer.step()
            optimizer.zero_grad()
                
        real.append(d.cpu())
        estimates.append(dd.detach().cpu())
    
    real = torch.cat(real, 0).squeeze()
    estimates = torch.cat(estimates, 0).squeeze()

    return compute_metrics(real, estimates, prefix='train_')


@torch.no_grad()
def evaluate(features, pivots, model, args):
    model.eval()

    real = []
    estimates = []

    steps = (args.accumulate // args.batch_size) * args.val_iterations
    for _ in trange(steps):
        o1, o2, d = prepare(features, pivots, args)
        
        emb1, emb2 = model(o1), model(o2)
        dd = torch.pow(emb1 - emb2, 2).sum(1, keepdim=True)
        
        real.append(d.cpu())
        estimates.append(dd.cpu())
    
    real = torch.cat(real, 0).squeeze()
    estimates = torch.cat(estimates, 0).squeeze()
    
    return compute_metrics(real, estimates)


@torch.no_grad()
def compute_metrics(real, estimates, prefix=''):
    errors = estimates - real
    
    abs_errors = errors.abs()
    rel_errors = (errors / (real + 1e-8)).abs()
    sq_errors = torch.pow(errors, 2)

    div = real / estimates

    return {
        prefix + 'mse': sq_errors.mean().item(),
        prefix + 'mse_std': sq_errors.std().item(),
        
        prefix + 'mape': rel_errors.mean().item(),
        prefix + 'mape_std': rel_errors.std().item(),
        
        prefix + 'mae': abs_errors.mean().item(),
        prefix + 'mae_std': abs_errors.std().item(),

        prefix + 'dist': (div.max() / div.min()).item(),
        prefix + 'mdist': (div / div.min()).mean().item(),
        prefix + 'mdist_std': (div / div.min()).std().item()
    }

 

def main(args):

    exp = expman.Experiment(args, root=args.rundir, ignore=('cuda', 'device', 'epochs', 'resume', 'rundir'))
    ckpt_dir = exp.path_to('ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    features = h5py.File(DATA_PATHS[args.data], 'r')['data']
    
    # a subset is (set, start, end)
    train_features = (features, 0, 750 * (10 ** 3))
    val_features = (features, 750 * (10 ** 3), 850 * (10 ** 3))
    # test_features = (features, 850 * (10 ** 3), 950 * (10 ** 3))

    pivots_start = 950 * (10 ** 3) + (4096 * args.pivot_seed)
    pivots = features[pivots_start:pivots_start + args.dim, :]
    pivots = torch.from_numpy(pivots).to(args.device)

    # Build the model
    model = get_model(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    # resume from checkpoint?
    if args.resume:
        ckpt = exp.ckpt('last.pth')
        assert os.path.exists(ckpt), "No checkpoint to resume from!"
        print('Resuming:', ckpt)
        
        ckpt = torch.load(ckpt)
        start_epoch = ckpt['metrics']['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Train loop
    progress = trange(start_epoch + 1, args.epochs + 1, initial=start_epoch, total=args.epochs)
    for epoch in progress:
        progress.set_description('TRAIN')
        train_metrics = train(train_features, pivots, model, optimizer, args)

        progress.set_description('EVAL')
        eval_metrics = evaluate(val_features, pivots, model, args)
        
        metrics = {'epoch': epoch, **train_metrics, **eval_metrics}        

        ckpt = exp.ckpt('last.pth')
        torch.save({
            'metrics': metrics,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, ckpt)

        if is_new_best(exp.log, metrics):
            best_ckpt = exp.ckpt(f'ckpt_e{epoch}.pth')
            shutil.copyfile(ckpt, best_ckpt)
            
        exp.push_log(metrics)        


def is_new_best(log, metrics):
    if log.empty:
        return True
    
    # any improvement on any of these metrics will trigger model snapshotting
    watch_metrics = ['mae', 'mape', 'mse']
    
    best = log[watch_metrics].min()
    current = pd.Series(metrics)[watch_metrics]
    return (current < best).any()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distance Estimation from Pivot Distances',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data params
    parser.add_argument('data', choices=('yfcc100m-hybridfc6'), help='dataset for training')
    parser.add_argument('-s', '--seed', type=int, default=23, help='Random initial seed')
    parser.add_argument('-p', '--pivot-seed', type=int, default=0, help='controls random selection of pivots')
    
    # Model params
    parser.add_argument('model', choices=('mlp', 'res-mlp'), help='dataset for training')
    parser.add_argument('-d', '--dim', type=int, default=200, help='Final dimensionality (also number of pivots)')
    parser.add_argument('-l', '--depth', type=int, default=2, help='Number of hidden layers (for MLP) or residual blocks (for ResMLP)')
    parser.add_argument('-n', '--batch-norm', action='store_true', default=False, help='Whether to use BN in MLP model')
    parser.add_argument('-o', '--dropout', type=float, default=0.0, help='Dropout probability for hidden layers')
    
    # Optimization params
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-b', '--batch-size', type=int, default=50, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('-i', '--iterations', type=int, default=20, help='Number of iterations (optimizer steps) per epoch')
    parser.add_argument('-v', '--val-iterations', type=int, default=20, help='Number of iterations (each of size defined by -a) for validation')
    parser.add_argument('-a', '--accumulate', type=int, default=1000, help='How many samples to accumulate before optimizer step (must be a multiple of batch size)')

    # Other
    parser.add_argument('-r', '--rundir', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('--resume', action='store_true', dest='resume', help='Resume training')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Run without CUDA')

    parser.set_defaults(cuda=True)
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    main(args)
