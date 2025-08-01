import os
import sys
import yaml
import time
import pandas as pd
from pathlib import Path
from hashlib import sha512
from itertools import product
from datetime import datetime
from simulator.Simulator import Simulator

def get_hyperparameters():
    """
    Fetches the hyperparameters from the docker compose config file
    :return: the experiment name and the hyperparameters (as a dictionary name -> values)
    """
    hyperparams = os.environ['LEARNING_HYPERPARAMETERS']
    hyperparams = yaml.safe_load(hyperparams)
    experiment_name, hyperparams = list(hyperparams.items())[0]
    return experiment_name.lower(), hyperparams

if __name__ == '__main__':

    total_experiments = 0

    datasets        = ['CIFAR100']
    clients         = 50
    batch_size      = 32
    local_epochs    = 2
    global_rounds   = 60
    data_dir        = 'data'
    max_seed        = 20

    data_output_directory = Path(data_dir)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    experiment_name, hyperparams = get_hyperparameters()
    a = hyperparams['algorithm'][0]

    if a == 0:
        algorithm = 'fedavg'
    elif a == 1:
        algorithm = 'fedproxy'
    elif a == 2:
        algorithm = 'Scaffold'
    elif a == 3:
        algorithm = 'ifca'

    csv_file = f'finished_experiment_log.csv'

    df = pd.DataFrame(columns=['timestamp', 'experiment'])

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        pass

    partitioning = 'Hard'
    areas = [3, 5, 9]
    iid_start = time.time()
    for seed in range(max_seed):
        for dataset in datasets:
            for area in areas:
                simulator = Simulator(algorithm, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
                simulator.seed_everything(seed)
                simulator.start(global_rounds)
                experiment_name = f'seed-{seed}_regions-{area}_algorithm_{algorithm}'
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_line = {'timestamp': timestamp, 'experiment': experiment_name}
                df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
                df.to_csv(csv_file, index=False)