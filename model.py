import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils.softargmax import SoftArgmax2D, create_meshgrid,SameSoftArgmax2D
from utils.preprocessing import augment_data, create_images_dict
from utils.image_utils import create_gaussian_heatmap_template, create_dist_mat, \
    preprocess_image_for_segmentation, pad, resize
from utils.dataloader import SceneDataset, scene_collate
from test import evaluate
from train import train
from for_real import r_test,e2e_test
from CoordConv import CoordConv2d,AddCoords
import cv2

class CoordAE(nn.Module):
    def __init__(self,obs_step,pred_step,num_scene,num_waypoint,embedding_size=64):
        super(CoordAE, self).__init__()
        self.softargmax_ = SoftArgmax2D(normalized_coordinates=False)
        self.encoder_pos = nn.Sequential(
            CoordConv2d(obs_step+num_scene, 32, kernel_size=3,padding=1, with_r=True),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, embedding_size, kernel_size=1, padding=0),
            nn.ELU(),
            CoordConv2d(embedding_size, embedding_size, 1, with_r=True),
            nn.Conv2d(embedding_size, embedding_size, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.decoder_end = nn.Sequential(
            nn.ConvTranspose2d(embedding_size+2, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(64, pred_step, kernel_size=1, padding=0),
            # nn.Sigmoid(),
        )
        self.decoder_traj = nn.Sequential(
            nn.ConvTranspose2d(embedding_size+2+num_waypoint, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            nn.Conv2d(64, pred_step, kernel_size=1, padding=0),
            # nn.Sigmoid(),
        )


    def pred_feature(self, in_data):
        feature = self.encoder_pos(in_data)
        return feature

    def pred_end(self, feature,w,h):
        addcoord_f = AddCoords(rank=2, w=w//2, h=h//2, skiptile=False)(feature)
        out = self.decoder_end(addcoord_f)
        return out

    def pred_traj(self, feature, waypoint,w,h):
        addcoord_f = AddCoords(rank=2, w=w//2, h=h//2, skiptile=False)(feature)
        encoder_out = torch.cat((addcoord_f, waypoint), dim=1)
        traj = self.decoder_traj(encoder_out)
        return traj

    def softmax(self, x):
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)

    def softargmax(self, output):
        return self.softargmax_(output)

    def sigmoid(self, output):
        return torch.sigmoid(output)

class TrajCoordAE:
    def __init__(self, obs_len, pred_len, params):

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.division_factor = 2 ** len(params['encoder_channels'])

        self.model = CoordAE(obs_step=obs_len,pred_step=pred_len,num_scene=params["semantic_classes"],num_waypoint=len(params["waypoints"]),embedding_size=32)



    def train(self, train_data, val_data, params, train_image_path, val_image_path, experiment_name, batch_size=8,
              num_goals=20, num_traj=1, device=None, dataset_name=None):
        """
        Train function
        :param train_data: pd.df, train data
        :param val_data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param train_image_path: str, filepath to train images
        :param val_image_path: str, filepath to val images
        :param experiment_name: str, arbitrary name to name weights file
        :param batch_size: int, batch size
        :param num_goals: int, number of goals per trajectory, K_e in paper
        :param num_traj: int, number of trajectory per goal, K_a in paper
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len

        print('Preprocess data')
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = '_mask.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = True
            # seg_mask = False

        # Load train images and augment train data and images
        df_train, train_images = augment_data(train_data, image_path=train_image_path, image_file=image_file_name,
                                              seg_mask=seg_mask)

        # Load val scene images
        val_images = create_images_dict(val_data, image_path=val_image_path, image_file=image_file_name)

        # Initialize dataloaders
        train_dataset = SceneDataset(df_train, resize=params['resize'], total_len=total_len)
        train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=scene_collate, shuffle=True)

        val_dataset = SceneDataset(val_data, resize=params['resize'], total_len=total_len)
        val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=scene_collate)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(train_images, factor=params['resize'], seg_mask=seg_mask)
        pad(train_images,division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocess_image_for_segmentation(train_images,classes=params["semantic_classes"])

        resize(val_images, factor=params['resize'], seg_mask=seg_mask)
        pad(val_images,division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet segmentation
        preprocess_image_for_segmentation(val_images,classes=params["semantic_classes"])

        model = self.model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])



        criterion = nn.BCEWithLogitsLoss()

        # Create template
        size = int(4200 * params['resize'])

        input_template = create_dist_mat(size=size)
        input_template = torch.Tensor(input_template).to(device)

        gt_template = create_gaussian_heatmap_template(size=size, kernlen=params['kernlen'], nsig=params['nsig'],
                                                       normalize=False)
        gt_template = torch.Tensor(gt_template).to(device)


        # self.train_Loss = []
        self.train_ADE = []
        self.train_FDE = []
        self.val_ADE = []
        self.val_FDE = []
        best_test_ADE = 99999999999999

        print('Start training')
        for e in tqdm(range(params['num_epochs']), desc='Epoch'):
            train_ADE, train_FDE = train(model, train_loader, train_images, e, obs_len, pred_len,
                                                     batch_size, params, gt_template, device,
                                                     input_template, optimizer, criterion, dataset_name, self.homo_mat)
            self.train_ADE.append(train_ADE)
            self.train_FDE.append(train_FDE)
            # self.train_Loss.append(train_loss)

            # For faster inference, we don't use TTST and CWS here, only for the test set evaluation
            val_ADE, val_FDE = evaluate(model, val_loader, val_images, num_goals, num_traj,
                                        obs_len=obs_len, batch_size=batch_size,
                                        device=device, input_template=input_template,
                                        waypoints=params['waypoints'], resize=params['resize'],
                                        temperature=params['temperature'], use_TTST=False,
                                        use_CWS=False, dataset_name=dataset_name,
                                        homo_mat=self.homo_mat, mode='val',exp_name=experiment_name)
            print(f'Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
            self.val_ADE.append(val_ADE)
            self.val_FDE.append(val_FDE)
            if val_ADE < best_test_ADE:
                print(f'Best Epoch {e}: \nVal ADE: {val_ADE} \nVal FDE: {val_FDE}')
                torch.save(model.state_dict(), 'save_model/' + experiment_name + '_weights.pt')
                best_test_ADE = val_ADE
            plt.plot(self.train_ADE, '-o')
            plt.plot(self.train_FDE, '-o')
            plt.xlabel('Epoch')
            plt.ylabel('Distance Error (Pixel)')
            plt.legend(['ADE', 'FDE'])
            plt.title('ADE vs FDE')
            plt.savefig(f"/home/ycchen/Desktop/ycchen_stuff/MyModel/save_image/{experiment_name}_Train_DE.png")
            plt.clf()
            plt.plot(self.val_ADE, '-o')
            plt.plot(self.val_FDE, '-o')
            plt.xlabel('Epoch')
            plt.ylabel('Distance Error (m)')
            plt.legend(['ADE', 'FDE'])
            plt.title('ADE vs FDE')
            plt.savefig(f"/home/ycchen/Desktop/ycchen_stuff/MyModel/save_image/{experiment_name}_Val_DE.png")
            plt.clf()



    def evaluate(self, data, params, image_path, exp_name,batch_size=8, num_goals=20, num_traj=1, rounds=1, device=None,
                 dataset_name=None):
        """
        Val function
        :param data: pd.df, val data
        :param params: dictionary with training hyperparameters
        :param image_path: str, filepath to val images
        :param batch_size: int, batch size
        :param num_goals: int, number of goals per trajectory, K_e in paper
        :param num_traj: int, number of trajectory per goal, K_a in paper
        :param rounds: int, number of epochs to evaluate
        :param device: torch.device, if None -> 'cuda' if torch.cuda.is_available() else 'cpu'
        :return:
        """

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_len = self.obs_len
        pred_len = self.pred_len
        total_len = pred_len + obs_len

        print('Preprocess data')
        dataset_name = dataset_name.lower()
        if dataset_name == 'sdd':
            image_file_name = '_mask.png'
        elif dataset_name == 'ind':
            image_file_name = 'reference.png'
        elif dataset_name == 'eth':
            image_file_name = 'oracle.png'
        else:
            raise ValueError(f'{dataset_name} dataset is not supported')

        # ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
        if dataset_name == 'eth':
            self.homo_mat = {}
            for scene in ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']:
                self.homo_mat[scene] = torch.Tensor(np.loadtxt(f'data/eth_ucy/{scene}_H.txt')).to(device)
            seg_mask = True
        else:
            self.homo_mat = None
            seg_mask = True

        test_images = create_images_dict(data, image_path=image_path, image_file=image_file_name)

        test_dataset = SceneDataset(data, resize=params['resize'], total_len=total_len)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=scene_collate)

        # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
        resize(test_images, factor=params['resize'], seg_mask=seg_mask)
        pad(test_images,division_factor=self.division_factor)  # make sure that image shape is divisible by 32, for UNet architecture
        preprocess_image_for_segmentation(test_images,classes=params["semantic_classes"])

        model = self.model.to(device)

        # Create template
        size = int(4200 * params['resize'])

        input_template = torch.Tensor(create_dist_mat(size=size)).to(device)

        self.eval_ADE = []
        self.eval_FDE = []

        print('Start testing')
        for e in tqdm(range(rounds), desc='Round'):
            test_ADE, test_FDE = evaluate(model, test_loader, test_images, num_goals, num_traj,
                                          obs_len=obs_len, batch_size=batch_size,
                                          device=device, input_template=input_template,
                                          waypoints=params['waypoints'], resize=params['resize'],
                                          temperature=params['temperature'], use_TTST=True,
                                          use_CWS=True if len(params['waypoints']) > 1 else False,
                                          rel_thresh=params['rel_threshold'], CWS_params=params['CWS_params'],
                                          dataset_name=dataset_name, homo_mat=self.homo_mat, mode='test',exp_name=exp_name)
            print(f'Round {e}: \nTest ADE: {test_ADE} \nTest FDE: {test_FDE}')

            self.eval_ADE.append(test_ADE)
            self.eval_FDE.append(test_FDE)

        print(
            f'\n\nAverage performance over {rounds} rounds: \nTest ADE: {sum(self.eval_ADE) / len(self.eval_ADE)} \nTest FDE: {sum(self.eval_FDE) / len(self.eval_FDE)}')

    def video_test(self, data, params, image_path, exp_name,input_video_path, num_goals=20, device=None,dataset_name=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_len = self.obs_len

        print('Preprocess data')
        image_file_name = 'oracle.png'

        self.homo_mat = None
        seg_mask = True

        test_images = create_images_dict(data, image_path=image_path, image_file=image_file_name)

        data["x"] = data["x"]*params['resize']
        data["y"] = data["y"]*params['resize']
        test_dataset = data

        resize(test_images, factor=params['resize'], seg_mask=seg_mask)
        pad(test_images,
            division_factor=self.division_factor)
        preprocess_image_for_segmentation(test_images)

        model = self.model.to(device)

        print('Start testing')
        r_test(model=model,data=test_dataset,val_images=test_images,num_goals=num_goals,obs_len=obs_len,device=device,waypoints=params['waypoints'],
               temperature=params['temperature'],exp_name=exp_name,input_video_path=input_video_path,resize=params["resize"])

    def end2end(self,params,input_video_path,frontend_arg, num_goals=20, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        obs_len = self.obs_len

        print('Preprocess data')
        mask = cv2.imread(f"{input_video_path}/mask.png",0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masks = {"mask": mask}

        self.homo_mat = None
        seg_mask = True

        resize(masks, factor=params['resize'], seg_mask=seg_mask)
        pad(masks,division_factor=self.division_factor)
        mask = masks["mask"]/255
        seg_1 = mask
        seg_2 = 1-mask
        mask = np.stack([seg_1,seg_2])

        model = self.model.to(device)
        e2e_test(model,mask,frontend_arg,temperature=params['temperature'],resize=params["resize"],waypoints=params['waypoints'],obs_len=obs_len,num_goals=num_goals,device=device)
        #https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch





    def load(self, path):
        print(self.model.load_state_dict(torch.load(path)))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
