import argparse
import h5py
import torch
import numpy as np
import pandas as pd
import os

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V
from torch.optim import Adam
from tqdm import trange


def prepare(features, pivots, args):

    b = args.batch_size
    d = features.shape[1]
    n = pivots.shape[0]

    o1 = np.random.choice(len(features), b, replace=False)
    o2 = np.random.choice(len(features), b, replace=False)

    o1 = torch.from_numpy(features[o1, ...]).cuda()
    o2 = torch.from_numpy(features[o2, ...]).cuda()

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
        mape = ((dd - d) / (d + 1e-8)).abs().mean()

        metrics = {
            'mse': mse.data.cpu()[0],
            'mape': mape.data.cpu()[0]
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

    avg_mse = 0
    avg_mape = 0
    n_samples = 0
    
    steps = (args.accumulate // args.batch_size) * args.iterations
    progress = trange(steps)
    for _ in progress:
        o1, o2, d = prepare(features, pivots, args)
        o1, o2, d = \
            V(o1, volatile=True).cuda(), \
            V(o2, volatile=True).cuda(), \
            V(d, volatile=True).cuda()

        emb1, emb2 = model(o1), model(o2)
        dd = torch.pow(emb1 - emb2, 2).sum(1, keepdim=True)
        mse = F.mse_loss(dd, d)
        mape = ((dd - d) / (d + 1e-8)).abs().mean()

        avg_mape += mape.data.cpu()[0]
        avg_mse += mse.data.cpu()[0]
        n_samples += d.shape[0]

        progress.set_postfix({'mape': avg_mape / n_samples, 'mse': avg_mse / n_samples})

    return {'mape': avg_mape / n_samples, 'mse': avg_mse / n_samples}


def main(args):

    # Prepare run dir
    params = vars(args)
    run_name = 'siamese_' \
               'a{0[accumulate]}_' \
               'b{0[batch_size]}_' \
               'd{0[dimensionality]}_' \
               'lr{0[lr]}_' \
               'i{0[iterations]}_' \
               'e{0[epochs]}'.format(params)

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

    train_features = features[:500 * 10 ** 3]
    val_features = features[500 * 10 ** 3: 750 * 10 ** 3]
    # test_features = features[750 * 10 ** 3:]

    pivots = h5py.File(args.pivots)['data']
    pivots = torch.from_numpy(pivots[:args.dimensionality]).cuda()

    model = nn.Sequential(
        nn.Linear(args.dimensionality, args.dimensionality),
        nn.ReLU(),
        nn.Linear(args.dimensionality, args.dimensionality),
    ).cuda()

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Train loop
    log = pd.DataFrame()
    train_log = pd.DataFrame()

    best = None
    progress = trange(1, args.epochs + 1)
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

        current = metrics['mse']
        if best is None or (current < best):
            best = current
            ckpt = os.path.join(ckpt_dir, 'ckpt_e{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'metrics': metrics,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distance Estimation from Pivot Distances')
    parser.add_argument('features', help='HDF5 file containing features')
    parser.add_argument('pivots', help='HDF5 file containing pivots')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('-i', '--iterations', type=int, default=20, help='Number of iterations (optimizer steps) per epoch')
    parser.add_argument('-b', '--batch-size', type=int, default=50, help='Batch size')
    parser.add_argument('-a', '--accumulate', type=int, default=1000, help='How many samples to accumulate before optimizer step (must be a multiple of batch size)')
    parser.add_argument('-d', '--dimensionality', type=int, default=200, help='Final dimensionality (also number of pivots)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-r', '--runs', type=str, default='runs/', help='Base dir for runs')
    args = parser.parse_args()

    main(args)
