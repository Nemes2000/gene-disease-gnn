# gene-disease-gnn

## Description

Exploring gene-disease associations is essential for a better understanding of complex diseases and for identifying potential diagnostic and therapeutic targets. In recent years, numerous machine learning methods have emerged, but they often struggle with data imbalance and fail to adequately account for complex relationships between diseases.
In my thesis, I investigated whether multi-task graph neural networks can improve the performance of gene-disease association prediction compared to a traditional single-task graph convolutional neural network. I used a special form of multi-task learning, called auxiliary learning, in which I tried to increase the accuracy of predictions by involving various auxiliary tasks in addition to a selected target task.
I selected several diseases as target tasks as binary classification problems, partly based on the performance of the base model and partly based on biological considerations. The selection of auxiliary tasks is crucial, as they greatly influence the outcome of the target task. I used different strategies to examine this: for example, selecting phenotypes belonging to the same disease group or diseases with a certain number of known gene associations. During training, I used an automatic weighting algorithm that adaptively modified the weight of the auxiliary tasks based on the correlation of the gradients associated with the losses of the target and auxiliary tasks.
I evaluated the results based on several metrics, including accuracy, sensitivity, F1 score, and the Area Under the Precision–Recall Curve (AUPRC).

The repo contains all code created during this project.

## Structure

```graphql
GENE-DISEASE-GNN/
│
├── data/                    # Datasets and graph data files
├── documents/               # Documentation, reports, or papers
├── notebooks/               # Jupyter notebooks for experiments
├── results/                 # Logs, metrics, and trained models
│
├── src/                     # Source code
│   ├── datasets/            # Data loading and preprocessing
│   ├── layers/              # Custom GNN and NN layers
│   ├── mappers/             # Mapping of gene/disease identifiers
│   ├── models/              # Model definitions
│   │   ├── basic_model.py
│   │   ├── lightning_gnn_model.py
│   │   ├── multitask_model.py
│   │   ├── utils.py
│   │   ├── weight_model.py
│   │   ├── config.py
│   │   └── hyperopt.py
│   ├── main.py              # Main experiment entry point
│   ├── see_fis.py           # Visualization and feature analysis
│   └── train.py             # Training script with CLI arguments
│
├── .gitignore
├── README.md
└── requirements.txt

```

## Code run

I used this repo with anaconda environment manager.

```bash
conda create -n gene-gnn python=3.12 -y
conda activate gene-gnn
```

```bash
pip install -r requirements.txt
```

### Command-line Arguments

| Argument           | Type        | Description                                                                                                        |
| ------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| `-model`           | `str`       | Specifies the model type. Choices: `basic`, `cls_weight`, or `multitask`. Default: `multitask`.                    |
| `-gnn_layer`       | `str`       | Type of GNN layer to use. Options: `GCN`, `GraphSAGE`. Default: `GraphSAGE`.                                       |
| `-model_ckpt_name` | `str`       | Name (or path) of the model checkpoint to load.                                                                    |
| `-disease`         | `str`       | Target disease identifier (e.g. C0000822). Use this at the `cls_weight` model training. Use the disease            |
| `-pr_disease`      | `str`       | Primary disease identifier used in multitask learning (e.g. C0000822). Use this at the `multitask` model training. |
| `-aux_diseases`    | `list[str]` | List of auxiliary diseases to include for multitask training. Example: `-aux_diseases C0001418 C0001430 C0001624`. |
| `--all_diseases`   | `flag`      | If set, all diseases (except the primary one) are used as auxiliary tasks.                                         |
| `-epoch`           | `int`       | Number of training epochs.                                                                                         |
| `--opt`            | `flag`      | If given, runs the model in hyperparameter optimization mode.                                                      |
| `-opt-step`        | `int`       | Number of optimization steps to perform if `--opt` is enabled.                                                     |
| `--test-dataset`   | `flag`      | Use a small test dataset for debugging or quick code testing.                                                      |
| `--new-dataset`    | `flag`      | Forces regeneration of the graph dataset before training.                                                          |

### Run a basic training

```bash
python src/main.py -epoch 100 -model basic -gnn_layer GCN --new-dataset
```

### Run a cls_weight training

```bash
python src/main.py -epoch 100 -model cls_weight -gnn_layer GCN -disease C1838979
```

### Run a multitask training

```bash
python src/main.py -epoch 100 -pr_disease C1838979 -aux_diseases C1838951 C0917798
```

### Run an optimalization

```bash
python src/main.py -epoch 100 --opt -opt-step 5 -pr_disease C1838979 -aux_diseases C1838951 C0917798
```

### Run a test on saved model

```bash
python src/main.py -model_ckpt_name /path/to/model
```

### Run feature importance score examination

Set dir path in src/see_fis.py line 10!

```bash
python src/see_fis.py
```

## Citation

Citation

If you use this project in your research, please cite this repository:

@misc{gene-disease-gnn,
author = {Nemes, Attila},
title = {Multi-task Learning in Graph Neural Networks (GNN)},
year = {2025},
url = {https://github.com/Nemes2000/gene-disease-gnn}
}

## License

This project is distributed under the MIT License.
See the LICENSE file for more details.
