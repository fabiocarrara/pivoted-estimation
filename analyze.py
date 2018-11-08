import argparse
import os
import glob2
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def _find_runs(base_dir, id_file='log.csv'):
    logs = glob2.glob(os.path.join(base_dir, '**', id_file))
    runs = [os.path.dirname(l) for l in logs]
    return runs


def dim(args):
    runs = _find_runs(args.runs)

    def _get_best(r):
        log = os.path.join(r, 'log.csv')
        log = pd.read_csv(log)
        best = log['mse'].min()
        best2 = log['mape'].min()
        
        params = os.path.join(r, 'params.csv')
        params = pd.read_csv(params)
        dim = params.loc[0, 'dimensionality']
        
        return dim, best, best2
        
    best = map(_get_best, runs)
    best = sorted(best)
    dim, best, best2 = zip(*best)
    
    best = pd.np.array(best)
    best = pd.np.sqrt(best)
    
    plt.semilogx(dim, best, marker='.', label='MSE')
    plt.semilogx(dim, best2, marker='.', color='red', label='MAPE')
    plt.xticks(dim, dim)
    plt.xlabel('Dim')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.savefig(args.output)
    

def train(args):
    runs = _find_runs(args.runs)

    def _get_log(r):
        params = os.path.join(r, 'params.csv')
        params = pd.read_csv(params)
        dim = params.loc[0, 'dimensionality']
        
        log = os.path.join(r, 'log.csv')
        log = pd.read_csv(log).set_index('epoch')
        col = log['mse'].rename(dim)
        
        return dim, col
        
    cols = map(_get_log, runs)
    cols = sorted(cols)
    dim, cols = zip(*cols)
    
    cols = pd.concat(cols, axis=1)
    cols.plot(y=list(dim))
    
    # best = pd.np.array(best)
    # best = pd.np.sqrt(best)
    
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.savefig(args.output)
    
    
if __name__ == '__main__':
    analyses = [k for k, x in globals().items() if callable(x) and not k.startswith('_')]
    
    parser = argparse.ArgumentParser(description='Analyzes Results')
    parser.add_argument('analysis', choices=analyses, help='which table to produce')
    parser.add_argument('-r', '--runs', default='runs/', help='directory containing runs')
    parser.add_argument('-o', '--output', help='where to save the result of the analysis')
    
    args = parser.parse_args()

    analysis_fn = args.analysis.replace('-', '_')
    analysis_fn = globals()[analysis_fn]
    analysis_fn(args)
