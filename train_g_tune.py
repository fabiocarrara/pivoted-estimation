import argparse
import h5py
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, ReduceLROnPlateau
from tqdm import trange

from hyperopt import hp
from hyperopt.pyll.base import scope

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from model_g import get_model_from_dict

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
            
            device = config.get('device', 'cpu')
            dim = config.get('dim', 64)
            batch_size = config.get('batch_size', 50)

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


    search_space = {
        'depth': hp.choice('depth', (1, 2, 4)),
        'fusion': hp.choice('fusion', ('early', 'mid', 'late')),
        'loss': hp.choice('loss', ('mse', 'sml1')),  # ('mse', 'mape', 'mae', 'sml1')
        'optim': hp.choice('optim', ('sgd', 'adam')),
        'lr': hp.loguniform('lr', -5, -1),
    }
    
    config = {
        'batch_size': 50,  # scope.int(hp.quniform('batch_size', 16, 512, 16)),
        'accumulate': 100,
        'iterations': 100,
        'val_iterations': 20,
        'batch_norm': True, #hp.choice('batch_norm', (True,)),
        'dropout': 0, # hp.choice('dropout', (0,)),
        'patience': 20,  # scope.int(hp.quniform('patience', 20, 100, 5)),        
        'lr_schedule': 'plateau', # hp.choice('lrschedule', ('plateau',)),
    }

    # pbt_search_space = {
    #     'lr': hp.loguniform('lr', -5, -1),
    # }

    name = f'n_pivots-{args.dim}'
    analysis = tune.run(regressor, name=name, local_dir=args.rundir,
                        num_samples=50,
                        search_alg=HyperOptSearch(search_space, metric="mape", mode='min'),
                        scheduler=ASHAScheduler(metric="mape", mode='min'),
                        checkpoint_freq=1,
                        checkpoint_score_attr='min-mape',
                        stop={"training_iteration": 100},
                        # scheduler=PopulationBasedTraining(
                        #    time_attr="training_iteration",
                        #    metric="mape", mode="min",
                        #    perturbation_interval=5,
                        #    hyperparam_mutations=pbt_search_space
                        #),
                        resources_per_trial={'cpu': 4, 'gpu': 0.33},
                        raise_on_failed_trial=False,
                        config=config,
                        resume=True,
                        )

    analysis.dataframe().to_csv(f'ray-{name}-results.csv')
    ckpt_dir = analysis.get_best_logdir('mape', mode='min')
    print(ckpt_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distance Estimation from Pivot Distances',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data params
    parser.add_argument('data', choices=('yfcc100m-hybridfc6',), help='dataset for training')
    parser.add_argument('dim', type=int, default=200, help='Final dimensionality (also number of pivots)')
    parser.add_argument('-s', '--seed', type=int, default=23, help='Random initial seed')
    
    # Other
    parser.add_argument('-r', '--rundir', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='Run without CUDA')

    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')

    main(args)
