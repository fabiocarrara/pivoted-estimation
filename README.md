# Learning Distance Estimators from Pivoted Embeddings of Metric Objects

Code to reproduce the results presented in 'Learning Distance Estimators from Pivoted Embeddings of Metric Objects'.

## How to reproduce

1. Get and prepare the data (instructions [here](data/yfcc100m-hybridfc6/README.md))

2. ``` pip install -r requirements.txt```

3. ``` python train_all.py yfcc100m-hybridfc6 ```

The `train_all.py` script adopts Ray Tune to handle and parallelize runs.
You should modify the script and adjust the parameters in `ray.init()` and `tune.run()` to your resources and desired parallelism.
