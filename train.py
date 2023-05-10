import torch
import torch.nn as nn
from utils.image_utils import get_patch,get_patch_grad
from tqdm import tqdm
import matplotlib.pyplot as plt




def train(model, train_loader, train_images, e, obs_len, pred_len, batch_size, params, gt_template, device, input_template, optimizer, criterion, dataset_name, homo_mat):
	"""
	Run training for one epoch

	:param model: torch model
	:param train_loader: torch dataloader
	:param train_images: dict with keys: scene_name value: preprocessed image as torch.Tensor
	:param e: epoch number
	:param params: dict of hyperparameters
	:param gt_template:  precalculated Gaussian heatmap template as torch.Tensor
	:return: train_ADE, train_FDE, train_loss for one epoch
	"""
	train_loss = 0
	train_ADE = []
	train_FDE = []
	model.train()
	# counter = 0
	# outer loop, for loop over each scene as scenes have different image size and to calculate segmentation only once
	for batch, (trajectory, meta, scene) in enumerate(tqdm(train_loader)):

		scene_image = train_images[scene].to(device).unsqueeze(0)
		model.train()

		# inner loop, for each trajectory in the scene
		for i in range(0, len(trajectory), batch_size):
			_, _, H, W = scene_image.shape  # image shape
			observed = trajectory[i:i+batch_size, :obs_len, :].to(device)
			observed_map = get_patch_grad(observed, H, W,device=device)
			observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])




			gt_future = trajectory[i:i + batch_size, obs_len:].to(device)
			gt_future_map = get_patch(gt_template, gt_future.reshape(-1, 2).cpu().numpy(), H, W)
			gt_future_map = torch.stack(gt_future_map).to(device).reshape([-1, pred_len, H, W])




			gt_waypoints = gt_future[:, params['waypoints']]
			gt_waypoint_map = get_patch_grad(gt_waypoints, H, W,device=device)
			gt_waypoint_map = torch.stack(gt_waypoint_map).reshape([-1, gt_waypoints.shape[1], H, W])

			# Concatenate heatmap and semantic map
			semantic_map = scene_image.expand(observed_map.shape[0], -1, -1, -1)  # expand to match heatmap size
			feature_input = torch.cat([semantic_map, observed_map], dim=1)


			# Forward pass
			# Calculate features
			features = model.pred_feature(feature_input)
			# features = model.pred_feature(observed_map,semantic_map)

			# Predict goal and waypoint probability distribution
			pred_goal_map = model.pred_end(features,w=W,h=H)
			goal_loss = criterion(pred_goal_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Prepare (downsample) ground-truth goal and trajectory heatmap representation for conditioning trajectory decoder
			gt_waypoints_maps_downsampled = nn.AvgPool2d(kernel_size=2, stride=2)(gt_waypoint_map)

			# Predict trajectory distribution conditioned on goal and waypoints
			pred_traj_map = model.pred_traj(features,gt_waypoints_maps_downsampled,w=W,h=H)
			traj_loss = criterion(pred_traj_map, gt_future_map) * params['loss_scale']  # BCEWithLogitsLoss

			# Backprop
			loss = goal_loss + traj_loss
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				train_loss += loss
				# Evaluate using Softargmax, not a very exact evaluation but a lot faster than full prediction
				pred_traj = model.softargmax(pred_traj_map)
				pred_goal = model.softargmax(pred_goal_map[:, -1:])

				# converts ETH/UCY pixel coordinates back into world-coordinates

				train_ADE.append(((((gt_future - pred_traj) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))
				train_FDE.append(((((gt_future[:, -1:] - pred_goal[:, -1:]) / params['resize']) ** 2).sum(dim=2) ** 0.5).mean(dim=1))

	train_ADE = torch.cat(train_ADE).mean()
	train_FDE = torch.cat(train_FDE).mean()

	return train_ADE.item(), train_FDE.item()