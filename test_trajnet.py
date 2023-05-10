import pandas as pd
import yaml
import argparse
import torch
from model import TrajCoordAE

name = 'zara2'
CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'
EXPERIMENT_NAME = f'{name}_trajnet'
DATASET_NAME = "eth"

TEST_DATA_PATH = f'data/eth_ucy/{name}_test.pkl'
TEST_IMAGE_PATH = 'data/eth_ucy'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 5  # K_e
NUM_TRAJ = 1  # K_a

ROUNDS = 1
BATCH_SIZE = 48



with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
# experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]


#### Load preprocessed Data



df_test = pd.read_pickle(TEST_DATA_PATH)



df_test.head()


#### Initiate model and load pretrained weights



model = TrajCoordAE(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)


model.load(f'save_model/{EXPERIMENT_NAME}_weights.pt')



#### Evaluate model



model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
               batch_size=BATCH_SIZE, rounds=ROUNDS,exp_name=EXPERIMENT_NAME,
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)