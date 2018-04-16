import argparse
import gzip
import h5py
import numpy as np

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm


def parse(args):
    i = 0
    data = h5py.File(args.output).create_dataset('data', (10**6, 4096), dtype='f', chunks=(512, 4096))
    with gzip.open(args.data, 'rb') as f:
        for line in tqdm(f, total=10**6):
            line = line.rstrip()
            if line:
                features = map(float, line.split(b'\t')[2:])
                features = np.fromiter(features, dtype=np.float32)
                features = features.reshape(1, -1)
                features = np.maximum(features, 0)
                features = normalize(features)
                data[i, ...] = features
                i += 1
                del features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse YFCC hybrid features textual data')
    parser.add_argument('data', help='File containing data')
    parser.add_argument('-o', '--output', default='features.h5', help='Output HDF5 file')
    args = parser.parse_args()

    parse(args)
