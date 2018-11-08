import argparse
import h5py
import torch
import numpy as np
import pandas as pd
import os
import glob2
import re

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V
from torch.optim import Adam
from tqdm import trange


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

    o1 = torch.from_numpy(features[o1[s1], ...][invs1]).cuda()
    o2 = torch.from_numpy(features[o2[s2], ...][invs2]).cuda()

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

    train_metrics = []
    
    steps_for_update = (args.accumulate // args.batch_size)
    steps = steps_for_update * args.iterations
    progress = trange(steps)
    for it in progress:
        o1, o2, d = prepare(features, pivots, args)
        o1, o2, d = \
            V(o1, requires_grad=False).cuda(), \
            V(o2, requires_grad=False).cuda(), \
            V(d, requires_grad=False).cuda()

        emb1, emb2 = model(o1), model(o2)
        dd = torch.pow(emb1 - emb2, 2).sum(1, keepdim=True)

        mse = F.mse_loss(dd, d)
        # mape = ((dd - d) / (d + 1e-8)).abs().mean()

        metrics = {
            'mse': '{:3.2f}'.format(mse.data.cpu()[0]),
            # 'mape': mape.data.cpu()[0]
        }
        progress.set_postfix(metrics)

        loss = mse
        loss.backward()

        if (it + 1) % steps_for_update:
            optimizer.step()
            train_metrics.append(metrics)
            optimizer.zero_grad()

    return train_metrics


def evaluate(features, pivots, model, args):
    model.eval()

    steps = (args.accumulate // args.batch_size) * args.val_iterations
    progress = trange(steps)
    
    real = []
    estimates = []
    
    for _ in progress:
        o1, o2, d = prepare(features, pivots, args)
        o1, o2, d = o1.cuda(), o2.cuda(), d.cuda()
        
        real.append(d)
        
        o1, o2, d = V(o1, volatile=True), \
                    V(o2, volatile=True), \
                    V(d, volatile=True)

        emb1, emb2 = model(o1), model(o2)
        dd = torch.pow(emb1 - emb2, 2).sum(1, keepdim=True)
        
        estimates.append(dd.data)
    
    real = torch.cat(real, 0).squeeze()
    estimates = torch.cat(estimates, 0).squeeze()

    if args.metrics:
        print('Couples evaluated:', real.shape[0])
    
    errors = estimates - real
    
    abs_errors = errors.abs()
    rel_errors = (errors / (real + 1e-8)).abs()
    sq_errors = torch.pow(errors, 2)

    div = real / estimates

    return {
        'mse': sq_errors.mean(),
        'mse_std': sq_errors.std(),
        
        'mape': rel_errors.mean(),
        'mape_std': rel_errors.std(),
        
        'mae': abs_errors.mean(),
        'mae_std': abs_errors.std(),

        'dist': div.max() / div.min(),
        'mdist': (div / div.min()).mean(),
        'mdist_std': (div / div.min()).std()
    }
    

def main(args):

    # Prepare run dir
    params = vars(args)
    run_name = 'siamese_' \
               'a{0[accumulate]}_' \
               'b{0[batch_size]}_' \
               'd{0[dimensionality]}_' \
               'l{0[layers]}_' \
               'do{0[dropout]}_' \
               'lr{0[lr]}_' \
               'i{0[iterations]}_' \
               'p{0[pivot_seed]}'.format(params)

    run_dir = os.path.join(args.runs, run_name)
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(ckpt_dir)

    log_file = os.path.join(run_dir, 'log.csv')
    train_log_file = os.path.join(run_dir, 'train_log.csv')

    param_file = os.path.join(run_dir, 'params.csv')
    pd.DataFrame(params, index=[0]).to_csv(param_file, index=False)

    features = h5py.File(args.features)['data']
    
    # a subset is (set, start, end)
    train_features = (features, 0, 750 * (10 ** 3))
    val_features = (features, 750 * (10 ** 3), 850 * (10 ** 3))
    # test_features = (features, 850 * (10 ** 3), 950 * (10 ** 3))

    pivots_start = 950 * (10 ** 3) + (4096 * args.pivot_seed)
    pivots = features[pivots_start:pivots_start + args.dimensionality, :]
    pivots = torch.from_numpy(pivots).cuda()

    # Build the model
    layers = [
        nn.Linear(args.dimensionality, args.dimensionality)
    ]
    
    for i in range(args.layers):
        layers.append(nn.ReLU())
        if args.dropout:
            layers.append(nn.Dropout(args.dropout))
        layers.append(nn.Linear(args.dimensionality, args.dimensionality))
        
    model = nn.Sequential(*layers).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Train loop
    log = pd.DataFrame()
    train_log = pd.DataFrame()

    watch_metrics = ['mae', 'mape', 'mse']
    best = pd.DataFrame({k: np.inf for k in watch_metrics}, index=[0])
    start_epoch = 0

    # resume from checkpoint
    if args.resume:
        ckpts = glob2.glob(os.path.join(ckpt_dir, '*.pth'))

        assert ckpts, "No checkpoints to resume from!"

        def get_epoch(ckpt_url):
            s = re.findall("ckpt_e(\d+).pth", ckpt_url)
            epoch = int(s[0]) if s else -1
            return epoch, ckpt_url

        start_epoch, ckpt = max(get_epoch(c) for c in ckpts)

        print('Resuming:', ckpt)

        ckpt = torch.load(ckpt)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        # compute metrics, print them, and exit
        if args.metrics:
            metrics = evaluate(val_features, pivots, model, args)
            metrics = pd.DataFrame(metrics, index=[0])
            print(metrics)
            return

        best = ckpt['metrics'][watch_metrics].sort_index(axis=1)

        log = pd.read_csv(log_file)
        train_log = pd.read_csv(train_log_file)

    progress = trange(start_epoch + 1, args.epochs + 1, initial=start_epoch, total=args.epochs)
    for epoch in progress:
        progress.set_description('TRAIN')
        metrics = train(train_features, pivots, model, optimizer, args)
        metrics = pd.DataFrame.from_records(metrics)
        metrics['epoch'] = epoch
        train_log = train_log.append(metrics, ignore_index=True)
        train_log.to_csv(train_log_file, index=False)

        progress.set_description('EVAL')
        metrics = evaluate(val_features, pivots, model, args)
        metrics['epoch'] = epoch
        log = log.append(metrics, ignore_index=True)
        log.to_csv(log_file, index=False)

        metrics = pd.DataFrame(metrics, index=[0]).sort_index(axis=1)
        current = metrics[watch_metrics].sort_index(axis=1)
        if (current < best).any(axis=1)[0]:
            best = current.where(current < best, best).sort_index(axis=1)
            ckpt = os.path.join(ckpt_dir, 'ckpt_e{}.pth'.format(epoch))
            torch.save({
                'metrics': metrics,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distance Estimation from Pivot Distances',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('features', help='HDF5 file containing features')
    parser.add_argument('-p', '--pivot-seed', type=int, default=0, help='controls random selection of pivots')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('-i', '--iterations', type=int, default=20, help='Number of iterations (optimizer steps) per epoch')
    parser.add_argument('-v', '--val-iterations', type=int, default=20, help='Number of iterations (each of size defined by -a) for validation')
    parser.add_argument('-b', '--batch-size', type=int, default=50, help='Batch size')
    parser.add_argument('-a', '--accumulate', type=int, default=1000, help='How many samples to accumulate before optimizer step (must be a multiple of batch size)')
    parser.add_argument('-d', '--dimensionality', type=int, default=200, help='Final dimensionality (also number of pivots)')
    parser.add_argument('-l', '--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('-o', '--dropout', type=float, default=0.0, help='Dropout probability for hidden layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-r', '--runs', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('--resume', action='store_true', dest='resume', help='Resume training')
    parser.add_argument('--metrics', action='store_true', dest='metrics', help='Compute metrics and exit (to be used with --resume)')
    parser.set_defaults(resume=False)
    parser.set_defaults(metrics=False)
    args = parser.parse_args()

    main(args)
