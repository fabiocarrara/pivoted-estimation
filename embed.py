import argparse
import h5py
import torch
import os

import torch.nn as nn

from torch.autograd import Variable as V
from tqdm import tqdm


def transform(objects, pivots):

    b, d = objects.shape
    n = pivots.shape[0]

    pivots = pivots.unsqueeze(0).expand(b, n, d)
    objects = objects.unsqueeze(1).expand(b, n, d)

    # pivot distances
    objects = torch.pow(objects - pivots, 2).sum(2)

    return objects


def embed(features, pivots, model, args):
    model.eval()
    
    embeds = h5py.File(args.embed, 'w')
    embeds = embeds.create_dataset('data', features.shape, dtype='f', chunks=(512, 4096))

    n_feats = features.shape[0]
    batch = args.batch_size
    with tqdm(total=n_feats) as pbar:
        for i in range(0, n_feats, batch):
            objects = torch.from_numpy(features[i: i+batch, ...]).cuda()
            objects = transform(objects, pivots)
            objects = V(objects, volatile=True)
            
            emb = model(objects).data.cpu()
            embeds[i: i+batch] = emb
            pbar.update(emb.shape[0])
    
    embeds.close()


def main(args):


    params = os.path.join(args.run, 'params.csv')
    params = pd.read_csv(params)
    dim = params.loc[0, 'dimensionality'] if 'dimensionality' in params.columns else 4096
    
    features = h5py.File(args.features)['data']
    pivots = h5py.File(args.pivots)['data']
    pivots = torch.from_numpy(pivots[:dim]).cuda()

    model = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    ).cuda()

    log = os.path.join(args.run, 'log.csv')
    log = pd.read_csv(log)
    idxmin = log['mse'].idxmin()
    
    checkpoint = os.path.join(args.run, 'ckpt', 'ckpt_e{}.pth'.format(idxmin)
    checkpoint = torch.load(args.run)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()

    embed(features, pivots, model, args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Distance Estimation from Pivot Distances')
    parser.add_argument('features', help='HDF5 file containing features')
    parser.add_argument('pivots', help='HDF5 file containing pivots')
    parser.add_argument('run', help='Trained model to use')
    parser.add_argument('embed', help='Output of embedded features')
    
    parser.add_argument('-b', '--batch-size', type=int, default=80, help='Batch size')
    args = parser.parse_args()

    main(args)
