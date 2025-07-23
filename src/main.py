import os
import sys
import yaml
import time
import pandas as pd
from pathlib import Path
from hashlib import sha512
from itertools import product
from simulator.Simulator import Simulator

if __name__ == '__main__':

    total_experiments = 0

    datasets        = ['MNIST', 'FashionMNIST', 'EMNIST']
    clients         = 50
    batch_size      = 32
    local_epochs    = 2
    global_rounds   = 30
    data_dir        = 'data-acsos-ae'
    max_seed        = 20

    data_output_directory = Path(data_dir)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    # Experiments IID
    partitioning = 'IID'
    experiment_name = 'fedavg'
    areas = 3
    iid_start = time.time()
    for seed in range(max_seed):
        seed_start = time.time()
        for dataset in datasets:
            simulator = Simulator(experiment_name, partitioning, areas, dataset, clients, batch_size, local_epochs, data_dir, seed)
            simulator.seed_everything(seed)
            simulator.start(global_rounds)
            total_experiments += 1
        seed_end = time.time()
        print(f'Seed {seed} took {seed_end - seed_start} seconds')
    iid_end = time.time()
    print(f'IID experiments took {iid_end - iid_start} seconds')

    # Experiments non-IID dirichlet
    partitioning = 'Dirichlet'
    experiment_names = ['fedavg', 'fedproxy', 'scaffold']
    areas = [3, 6, 9]
    non_iid_start = time.time()
    for seed in range(max_seed):
        seed_start = time.time()
        for experiment_name in experiment_names:
            for dataset in datasets:
                for area in areas:
                    print(f'starting dirichlet seed {seed} experiment {experiment_name} dataset {dataset} area {area}')
                    simulator = Simulator(experiment_name, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
                    simulator.seed_everything(seed)
                    simulator.start(global_rounds)
                    total_experiments += 1
        seed_end = time.time()
        print(f'Seed {seed} took {seed_end - seed_start} seconds')
    non_iid_end = time.time()
    print(f'non-IID experiments took {non_iid_end - non_iid_start} seconds')

    # Experiments non-IID hard EMNIST
    partitioning = 'Hard'
    experiment_names = ['fedavg', 'fedproxy', 'scaffold']
    areas = [3, 5, 9]
    non_iid_start = time.time()
    for seed in range(max_seed):
        seed_start = time.time()
        for experiment_name in experiment_names:
            for dataset in ['EMNIST']:
                for area in areas:
                    print(f'starting hard seed {seed} experiment {experiment_name} dataset {dataset} area {area}')
                    simulator = Simulator(experiment_name, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
                    simulator.seed_everything(seed)
                    simulator.start(global_rounds)
                    total_experiments += 1
        seed_end = time.time()
        print(f'Seed {seed} took {seed_end - seed_start} seconds')
    non_iid_end = time.time()
    print(f'non-IID experiments hard EMNIST took {non_iid_end - non_iid_start} seconds')

    # Experiments non-IID hard MNIST and Fashion
    partitioning = 'Hard'
    experiment_names = ['fedavg', 'fedproxy', 'scaffold']
    areas = [3]
    non_iid_start = time.time()
    for seed in range(max_seed):
        seed_start = time.time()
        for experiment_name in experiment_names:
            for dataset in ['MNIST', 'FashionMNIST']:
                for area in areas:
                    print(f'starting hard seed {seed} experiment {experiment_name} dataset {dataset} area {area}')
                    simulator = Simulator(experiment_name, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
                    simulator.seed_everything(seed)
                    simulator.start(global_rounds)
        seed_end = time.time()
        print(f'Seed {seed} took {seed_end - seed_start} seconds')
    non_iid_end = time.time()
    print(f'non-IID experiments hard MNIST and Fashion took {non_iid_end - non_iid_start} seconds')
    #
    #
    print(f'Total experiments {total_experiments}')