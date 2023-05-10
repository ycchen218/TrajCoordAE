import os
import yaml
from model import TrajCoordAE
import pandas as pd



name = 'zara2'
CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = f'{name}_trajnet'
DATASET_NAME = "eth"

TEST_DATA_PATH = f'data/eth_ucy/{name}_test.pkl'
TEST_IMAGE_PATH = 'data/eth_ucy'
OBS_LEN = 8
PRED_LEN = 12
NUM_GOALS = 3

args = {
    'input_path': './e2e/Sample.mp4',
    'save_path': 'output/',
    'frame_interval': 2,
    'fourcc': 'mp4v',
    'device': '',
    'save_txt': 'output/predict/',
    'display': True,
    'display_width': 800,
    'display_height': 600,
    'cam': -1,
    'weights': './DeepSORT_YOLOv5_Pytorch-master/yolov5/weights/yolov5s.pt',
    'img_size': 640,
    'conf_thres': 0.5,
    'iou_thres': 0.5,
    'classes': [0],
    'agnostic_nms': True,
    'augment': True,
    'config_deepsort': './DeepSORT_YOLOv5_Pytorch-master/configs/deep_sort.yaml'
}

import torch
print(torch.cuda.is_available())

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

df_test = pd.read_pickle(TEST_DATA_PATH)
df_test.head()

model = TrajCoordAE(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
model.load(f'save_model/{EXPERIMENT_NAME}_weights.pt')


input_video_path = f'e2e'
model.end2end(params,frontend_arg=args,num_goals=NUM_GOALS,device="cuda",input_video_path=input_video_path)
# input_video_path = f'./video/{name}_video.avi'
# input_video_path = f'./DeepSORT_YOLOv5_Pytorch-master/video/{name}_video.avi'
# model.video_test(df_test,params,image_path=TEST_IMAGE_PATH,exp_name=EXPERIMENT_NAME,num_goals=NUM_GOALS,device="cuda",dataset_name=DATASET_NAME,input_video_path=input_video_path)