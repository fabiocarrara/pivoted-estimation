# Getting YFCC100M-HNfc6 features

In our experiments, we adopt a 1M subset of the YFCC100M-HNfc6 deep features dataset.

Follow the instruction on http://www.deepfeatures.org/download.html to access the dataset.

Download `YFCC100M_hybridCNN_gmean_fc6_1.txt.gz` and parse it into an HDF5 file:
```bash
python parse_data.py YFCC100M_hybridCNN_gmean_fc6_1.txt.gz -o features-001-of-100.h5
```
