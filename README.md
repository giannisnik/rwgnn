# Random Walk Graph Neural Networks

This repository is the official implementation of [Random Walk Graph Neural Networks](https://proceedings.neurips.cc/paper/2020/file/ba95d78a7c942571185308775a97a3a0-Paper.pdf). 

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

|  Model name  |     MUTAG    |     D&D      |     NCI1     |   PROTEINS   |    ENZYMES   |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|      SP      | 80.2 (± 6.5) | 78.1 (± 4.1) | 72.7 (± 1.4) | 75.3 (± 3.8) | 38.3 (± 8.0) |
|      GR      | 80.8 (± 6.4) | 75.4 (± 3.4) | 61.8 (± 1.7) | 71.6 (± 3.1) | 25.1 (± 4.4) |
|      WL      | 84.6 (± 8.3) | 78.1 (± 2.4) | 84.8 (± 2.5) | 73.8 (± 4.4) | 50.3 (± 5.7) |
|     DGCNN    | 84.0 (± 6.7) | 76.6 (± 4.3) | 76.4 (± 1.7) | 72.9 (± 3.5) | 38.9 (± 5.7) |
|   DiffPool   | 79.8 (± 7.1) | 75.0 (± 3.5) | 76.9 (± 1.9) | 73.7 (± 3.5) | 59.5 (± 5.6) |
|     ECC      | 75.4 (± 6.2) | 72.6 (± 4.1) | 76.2 (± 1.4) | 72.3 (± 3.4) | 29.5 (± 8.2) |
|     GIN      | 84.7 (± 6.7) | 75.3 (± 2.9) | 80.0 (± 1.4) | 73.3 (± 4.0) | 59.6 (± 4.5) |
|  GraphSAGE   | 83.6 (± 9.6) | 72.9 (± 2.0) | 76.0 (± 1.8) | 73.0 (± 4.5) | 58.2 (± 6.0) |
| 1-step RWNN  | 89.2 (± 4.3) | 77.6 (± 4.7) | 71.4 (± 1.8) | 74.7 (± 3.3) | 56.7 (± 5.2) |
| 2-step RWNN  | 88.1 (± 4.8) | 76.9 (± 4.6) | 73.0 (± 2.0) | 74.1 (± 2.8) | 57.4 (± 4.9) |
| 3-step RWNN  | 88.6 (± 4.1) | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |


| Model name  | IMDB-BINARY  |  IMDB-MULTI  | REDDIT-BINARY | REDDIT-MULTI-5K |   COLLAB     |
| ----------- | ------------ | ------------ | ------------- | --------------- | ------------ |
|     SP      | 57.7 (± 4.1) | 39.8 (± 3.7) | 89.0 (± 1.0)  |  51.1 (± 2.2)   | 79.9 (± 2.7) |
|     GR      | 63.3 (± 2.7) | 39.6 (± 3.0) | 76.6 (± 3.3)  |  38.1 (± 2.3)   | 71.1 (± 1.4) |
|     WL      | 72.8 (± 4.5) | 51.2 (± 6.5) | 74.9 (± 1.8)  |  49.6 (± 2.0)   | 78.0 (± 2.0) |
|    DGCNN    | 69.2 (± 3.0) | 45.6 (± 3.4) | 87.8 (± 2.5)  |  49.2 (± 1.2)   | 71.2 (± 1.9) |
|   DiffPool  | 68.4 (± 3.3) | 45.6 (± 3.4) | 89.1 (± 1.6)  |  53.8 (± 1.4)   | 68.9 (± 2.0) |
|     ECC     | 67.7 (± 2.8) | 43.5 (± 3.1) |      OOR      |       OOR       |      OOR     |
|     GIN     | 71.2 (± 3.9) | 48.5 (± 3.3) | 89.9 (± 1.9)  |  56.1 (± 1.7)   | 75.6 (± 2.3) |
|  GraphSAGE  | 68.8 (± 4.5) | 47.6 (± 3.5) | 84.3 (± 1.9)  |  50.0 (± 1.3)   | 73.9 (± 1.7) |
| 1-step RWNN | 70.8 (± 4.8) | 47.8 (± 3.8) | 90.4 (± 1.9)  |  51.7 (± 1.5)   | 71.7 (± 2.1) |
| 2-step RWNN | 70.6 (± 4.4) | 48.8 (± 2.9) | 90.3 (± 1.8)  |  51.7 (± 1.4)   | 71.3 (± 2.1) |
| 3-step RWNN | 70.7 (± 3.9) | 47.8 (± 3.5) | 89.7 (± 1.2)  |  53.4 (± 1.6)   | 71.9 (± 2.5) |


### Cite
Please cite our paper if you use this code:
```
@inproceedings{nikolentzos2020random,
  title={Random Walk Graph Neural Networks},
  author={Nikolentzos, Giannis and Vazirgiannis, Michalis},
  booktitle={Proceedings of the 34th Conference on Neural Information Processing Systems},
  pages={16211--16222},
  year={2020}
}
```
