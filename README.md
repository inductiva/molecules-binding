[![Python package](https://github.com/inductiva/molecules-binding/actions/workflows/python-package.yml/badge.svg)](https://github.com/inductiva/molecules-binding/actions/workflows/python-package.yml)

# Geometric Deep Learning for Molecular Interactions

This repository contains the source code and documentation related with Sofia Guerreiro's 
research on Geometric Deep Learning for molecules binding.

MSc Thesis title (Instituto Superior Técnico), Nov. 2023 - Sofia Guerreiro:

"Predicting Protein-Ligand Binding Affinity using Graph Neural Networks"

# Running the code

This code was developed using Python version 3.10.12. It is recommended to use the same or a compatible version of Python.

If you don't already have Python 3.10.12 installed, you can download it from the [official Python website](https://www.python.org/downloads/release/python-31012/).


## Mlflow

We are using
[mlflow](https://mlflow.org/docs/latest/python_api/mlflow.html) to
keep track of our experiments. We have our own remote server to which
we can log everything from any computer which was created following
the instructions in [this
tutorial](https://towardsdatascience.com/managing-your-machine-learning-experiments-with-mlflow-1cd6ee21996e).

This is a nice feature to have but it is not required in any way to
run our code. Everything will be logged locally by default. That is,
all experiments will be logged to the folder `mlruns` created in the
directory from which the script is launched. To then look at the
experiments in the browser we just need to run the command `mlflow
ui`.

## Installing everything

The next step is to actually clone the repository using:

```bash
git clone git@github.com:inductiva/molecules-binding.git
```

The very next step is to create a virtual environment. This will solve
any clashes with the library versions used here and anything else that
might be installed in your own system:

```bash
python3 -m venv .env
source .env/bin/activate
```

After creating and activating the virtual environment we can install
all the requirements of the project using:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Next, because `molecules-binding` is actually packaged we can install it using:

```bash
pip install -e .
```

## Processing the dataset

We included a small example dataset, in the directory `example_dataset`. 
For using a real dataset, you can download [PDBBind](http://www.pdbbind.org.cn/).
In this project, the majority of the experiments included PDBBind general set 2016.
To process the dataset and store it, first create a directory where you want to keep
stored datasets (e.g., `/datasetsprocessed/`), and then run the script
`process_dataset_interaction.py`. For instance,

```bash
python scripts/process_dataset_interaction.py --affinity_dir=example_dataset/index/INDEX_general_PL_data.2020 --data_dir=example_dataset/ --path_dataset=../datasetsprocessed/example_data_processed --threshold=8 --which_file_ligand="mol2" --not_include_test_set=True --separate_edges=False
```

## Training the model
To train a model is simply a matter of running the script `train_graphnet_lightning.py`:

```bash
python scripts/train_graphnet_lightning.py --path_dataset=../datasetsprocessed/example_data_processed --dropout_rate=0.1 --max_epochs=2500 --use_gpu=True --batch_size=3 --num_hidden_linear=256,256 --train_split=0.9 --learning_rate=0.0001 --weight_decay=0.0001 --use_batch_norm=True --comment="running final architecture" --embedding_layers=128,128 --use_message_passing=True --which_gnn_model=NodeEdgeGNN --size_processing_steps=128 --early_stopping_patience=500 --num_processing_steps=3 --splitting_seed=24 --save_model=False --mlflow_server_uri=<Your mlflow server uri>
```

You can choose other flags, with different parameters.

## Evaluating the model on a test set

If you choose to save the model, you can later evaluate the performance on other test sets. You can create a directory to store the results (e.g., `/results/`) and rin the script `evaluate_model.py` 

```bash
python scripts/evaluate_model.py --path_dataset=<Path to the processed dataset to test> --mlflow_server_uri=<your mlflow server ui> --results_dir=../results --run_id=<The run ID where the model was trained>
```
