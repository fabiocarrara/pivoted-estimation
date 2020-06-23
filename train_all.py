import argparse
import h5py
import math
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, ReduceLROnPlateau
from tqdm import trange

import ray
from ray import tune

from model import get_model_from_dict

# paths to HDF5 file containing features
DATA_PATHS = {
    'yfcc100m-hybridfc6': 'data/yfcc100m-hybridfc6/features-001-of-100.h5'
}

@torch.no_grad()
def prepare(features, args):

    features, start, end = features
    n_features = end - start

    n = args.get('dim', 64)  # number of pivots
    b = args.get('batch_size', 50)
    d = features.shape[1]  # dimensionality of original features

    # takes n features as pivots and two batches of features as data points
    n_take = n + 2*b

    x = np.random.randint(n_features - n_take)
    x = features[x:x + n_take, ...]
    np.random.shuffle(x)

    x = torch.from_numpy(x).to(args.get('device', 'cpu'))
    o1, o2, pivots = x.split((b, b, n))

    # object-object euclidean distances (target)
    oo = torch.pow(o1 - o2, 2).sum(1, keepdim=True).sqrt()

    # object-pivot euclidean distances (input)
    op1 = torch.cdist(o1, pivots, p=2)
    op2 = torch.cdist(o2, pivots, p=2)

    # pivot-pivot distances (additional input)
    pp = torch.pdist(pivots, p=2)

    return op1, op2, pp, oo



def train(data, model, optimizer, scheduler, args):
    model.train()
    optimizer.zero_grad()

    real = []
    estimates = []

    steps_for_update = (args.get('accumulate', 100) // args.get('batch_size',  50))
    steps = steps_for_update * args.get('iterations', 100)
    progress = trange(steps, disable=args.get('no_progress', True))
    for it in progress:
        op1, op2, pp, oo = prepare(data, args)  # already moved to device

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

        loss_type = args.get('loss', 'mse')
        if loss_type == 'mse':
            mse.backward()
        elif loss_type == 'mae':
            mae.backward()
        elif loss_type == 'mape':
            mape.backward()
        elif loss_type == 'sml1':
            sml1.backward()

        if (it + 1) % steps_for_update:
            optimizer.step()
            optimizer.zero_grad()
            if args.get('lr_schedule', 'fixed') == 'cycle':
                scheduler.step()

        real.append(oo)
        estimates.append(oo_p.detach())

    real = torch.cat(real, 0).squeeze()
    estimates = torch.cat(estimates, 0).squeeze()

    return compute_metrics(real, estimates, prefix='train_')


@torch.no_grad()
def evaluate(data, model, args):
    model.eval()

    real = []
    estimates = []

    steps = (args.get('accumulate', 100) // args.get('batch_size', 50)) * args.get('val_iterations', 20)
    for _ in trange(steps, disable=args.get('no_progress', True)):
        op1, op2, pp, oo = prepare(data, args)

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
    features = DATA_PATHS[args.data]

    class regressor(tune.Trainable):
        def _setup(self, config):
            self.config = config

            device = config.get('device', 'cuda')
            dim = config.get('dim', 64)
            batch_size = config.get('batch_size', 50)

            config['fusion'], config['depth'] = config.get('architecture', ('early', 1))
            # Build the model
            self.model = get_model_from_dict(config).to(device)

            lr = config.get('lr', 0.001)
            # Optimizer
            optim = config.get('optim', 'sgd')
            if optim == 'sgd':
                self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
            elif optim == 'adam':
                self.optimizer = Adam(self.model.parameters(), lr=lr)

            # LR Scheduler
            lr_schedule = config.get('lr_schedule', 'fixed')
            if lr_schedule == 'fixed':
                self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # no-op scheduler
            elif lr_schedule == 'cycle':
                self.scheduler = CyclicLR(self.optimizer, base_lr=(lr / 100), max_lr=lr, mode='triangular2')
            elif lr_schedule == 'plateau':
                self.scheduler = ReduceLROnPlateau(self.optimizer, patience=config.get('patience'))

        def _train(self):
            metrics = {}

            # Prepare Data
            features = h5py.File(DATA_PATHS[args.data], 'r')['data']
            train_data = (features, 0, 750 * (10 ** 3))
            val_data = (features, 750 * (10 ** 3), 900 * (10 ** 3))

            train_metrics = train(train_data, self.model, self.optimizer, self.scheduler, self.config)
            test_metrics = evaluate(val_data, self.model, self.config)

            metrics.update(train_metrics)
            metrics.update(test_metrics)

            if self.config.get('lr_schedule', 'fixed') == 'plateau':
                loss = self.config.get('loss', 'mse')
                self.scheduler.step(metrics[loss])

            return metrics

        def _save(self, checkpoint_dir):
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }

            checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(checkpoint, checkpoint_path)
            return checkpoint_path

        def _restore(self, checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])


    grid_search_space = {
        'dim': tune.grid_search([2, 4, 8, 16, 32, 64, 128]), #, 256, 512, 1024, 2048, 4096]),
        'architecture': tune.grid_search([
            ('early', 1),
            ('early', 2),
            ('early', 4),
            ('mid', 2),
            ('mid', 4),
            ('late', 1),
            ('late', 2),
            ('late', 4)
        ])
    }

    config = {
        'device': args.device,
        'batch_size': 50,
        'accumulate': 100,
        'iterations': 100,
        'val_iterations': 20,
        'batch_norm': True,
        'dropout': 0,
        'patience': 20,
        'lr_schedule': 'plateau',
        'loss': 'sml1',
        'optim': 'sgd',
        'lr': 0.05,
        **grid_search_space
    }

    ray.init(num_cpus=6, num_gpus=1, memory=6*1024**3, object_store_memory=6*1024**3, redis_max_memory=2*1024**3,
             include_webui=False)

    def stop_condition(trial_id, report):
        return math.isnan(report['mape']) or report['training_iteration'] > 100

    analysis = tune.run(regressor, name='arch-search', local_dir=args.rundir,
                        keep_checkpoints_num=1, checkpoint_score_attr='min-mape', checkpoint_freq=1,
                        stop=stop_condition,
                        resources_per_trial={'cpu': 6, 'gpu': 1},
                        raise_on_failed_trial=False,
                        with_server=False, queue_trials=True,
                        config=config, resume=args.resume)

    analysis.dataframe().to_csv(f'results.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distance Estimation from Pivot Distances',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data params
    parser.add_argument('data', choices=('yfcc100m-hybridfc6',), help='dataset for training')
    parser.add_argument('-s', '--seed', type=int, default=23, help='Random initial seed')

    # Other
    parser.add_argument('-r', '--rundir', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('--resume', action='store_true', help='Resume ray job')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Run without CUDA')

    parser.set_defaults(cuda=True, resume=False)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    main(args)
