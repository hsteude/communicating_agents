# Communicating Agents
### Re-Implementation of "Operationally meaningful representations of physical systems in neural networks"
This repo is our implementation of the NN architecture proposed in H. Poulsen Nautrup, T. Metger, R. Iten, S. Jerbi, L.M. Trenkwalder, H.Wilming, H.J. Briegel, and R. Renner. "Operationally meaningful representations of physical systems in neural networks" (2020).
Note that the original implementation can already be found on [github](https://github.com/tonymetger/communicating_scinet), we implemented this one from scratch and chose different approaches e.g. with regards to the training data generation or the model implementation.

#### Project structure

```
.
├── README.md                                           --> this file
├── comm_agents                                         --> package 
│   ├── __init__.py
│   ├── data                                            --> data set related code
│   │   ├── __init__.py
│   │   ├── data_generator.py
│   │   ├── data_handler.py
│   │   ├── optimal_answers.py
│   │   └── reference_experiments.py
│   ├── models                                          --> model implementations and trainng
│   │   ├── __init__.py
│   │   ├── model_multi_enc.py
│   │   ├── model_single_enc.py
│   │   ├── trainig_multi_enc.py
│   │   └── trainig_single_enc.py
│   └── utils.py
├── config.json                                         --> run configurations
├── data                                                --> data files 
│   └── training
├── lightning_logs                                      --> model logs (e.g. for tensorboard)
├── models                                              --> model directory
├── notebooks                                           --> for visualizations
│   ├── 01_reference_experiment_visualization.ipynb
│   ├── 02_generated_data_exploration.ipynb
│   ├── 03_single_enc_analysis-1.ipynb
│   └── 04_multi_enc_analysis.ipynb
└── setup.py
```

#### Installation
Assuming you have a fresh virtual environment wiht python 3.7 or higher, run the following code in the root of this project.
```shell
pip install -e .
```

#### Run the proejct
Data set generation (with virt env activated):
```shell
python comm_agents/data/data_generator.py
```
Note that this one runs quite a while and it makes sense to use a beefy machine (many cpus) and specify the number of jobs to run in the config.json (e.g. 48).

Model training:
```shell
python comm_agents/models/trainig_single_enc.py
```







