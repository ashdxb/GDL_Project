## GIRL Implementation

### Requirements
* `python==3.8.*`
* `requirements.txt`

Just run the code in: `run_many_exp.py`, run the datasets and configurations you want.
## KR-loss estimation on synthetic data
* to reproduce the graphs in the Kernel Alternatives section, of the paper simply run the kernels_test.py file

## Graph Barlow Twins
to reproduce the results of the Graph Barlow Twins follow these instructions:
1. Clone the repository of the [Graph Barlow Twins](https://github.com/pbielak/graph-barlow-twins.git)
2. create and activate virtual environment
3. install the requirements from the requirements.txt file:
```bash
pip install -r requirements.txt
```
4. from the root of the repository run the following command to preprocess the dataset:
```bash
PYTHONPATH=. python3 experiments/scripts/preprocess_dataset.py PPI
```
5. replce the PPI part of the config file "experiments\configs\batched" with this:
```
  PPI: <<
  :
  *default
  encoder_cls: "gssl.batched.encoders.BatchedGAT"
  batch_sizes: [
    512,
    1024
  ]
  total_epochs: 100
  warmup_epochs: 50
  log_interval: 100
  num_splits: 5
  emb_dim: 512
  lr_base: 5.e-3
```
6. run the following command to train the model:
```bash
PYTHONPATH=. python3 experiments/scripts/batched/train.py PPI
```
