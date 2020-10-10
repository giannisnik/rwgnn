# Random Walk Graph Neural Networks

This repository is the official implementation of [Random Walk Graph Neural Networks](http://www.lix.polytechnique.fr/Labo/Ioannis.Nikolentzos/files/rw_gnns_neurips20). 

## Requirements

Code is written in Python 3.6 and requires:
* PyTorch 1.5
* scikit-learn 0.21

### Datasets
Use the following link to download datasets: 
```
https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
```
Extract the datasets into the `datasets` folder.

## Training and Evaluation

To train and evaluate the model in the paper, run this command:

```
python main.py --dataset <dataset_name> 
```

## Example

To train and evaluate the model on MUTAG, first specify the hyperparameters in the `main.py` file and then run:

```
python main.py --dataset MUTAG --use-node-labels
```

## Results

Our model achieves the following performance on standard graph classification datasets (note that we used the evaluation procedure and same data splits as in [this paper](https://openreview.net/pdf?id=HygDF6NFPB)):

| Model name  |     MUTAG    |      D&D     |     NCI1     |   PROTEINS   |    ENZYMES   |
| ------------|-----------------------------------------------------------|------------- |
| 1-step RWNN | 89.2 (± 4.3) | 77.6 (± 4.7) | 71.4 (± 1.8) | 74.7 (± 3.3) | 56.7 (± 5.2) |
| 2-step RWNN | 88.1 (± 4.8) | 76.9 (± 4.6) | 73.0 (± 2.0) | 74.1 (± 2.8) | 57.4 (± 4.9) |
| 3-step RWNN | 88.6 (± 4.1) | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |


| Model name  | IMDB-BINARY  |  IMDB-MULTI  | REDDIT-BINARY | REDDIT-MULTI-5K |   COLLAB     |
| ------------|----------------------------------------------------------------------------- |
| 1-step RWNN | 70.8 (± 4.8) | 47.8 (± 3.8) | 90.4 (± 1.9)  |  51.7 (± 1.5)   | 71.7 (± 2.1) |
| 1-step RWNN | 70.6 (± 4.4) | 48.8 (± 2.9) | 90.3 (± 1.8)  |  51.7 (± 1.4)   | 71.3 (± 2.1) |
| 1-step RWNN | 70.7 (± 3.9) | 47.8 (± 3.5) | 89.7 (± 1.2)  |  53.4 (± 1.6)   | 71.9 (± 2.5) |


### Cite
Please cite our paper if you use this code:
```
@inproceedings{nikolentzos2020message,
  title={Message Passing Attention Networks for Document Understanding},
  author={Nikolentzos, Giannis and Tixier, Antoine Jean-Pierre and Vazirgiannis, Michalis},
  booktitle={Proceedings of the 34th AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
