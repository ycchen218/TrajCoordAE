import pandas as pd
import yaml
import argparse
import torch
from model import TrajCoordAE


name = 'zara2'
CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = f'{name}_trajnet'  # arbitrary name for this experiment
DATASET_NAME = 'eth'

TRAIN_DATA_PATH = f'data/eth_ucy/{name}_train.pkl'
TRAIN_IMAGE_PATH = 'data/eth_ucy'
VAL_DATA_PATH = f'data/eth_ucy/{name}_test.pkl'
VAL_IMAGE_PATH = 'data/eth_ucy'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 6

print(f"Now training the {name} data")

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]



df_train = pd.read_pickle(TRAIN_DATA_PATH)
df_val = pd.read_pickle(VAL_DATA_PATH)



df_train.head()




model = TrajCoordAE(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)



model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ,
            device=None, dataset_name=DATASET_NAME)