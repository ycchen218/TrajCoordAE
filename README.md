# Trajectory Prediction by CoordAutoencoder
## Train & Test
### requirement
opencv-contrib-python=4.7.0.72<br>
pytorch=1.13.1=py3.8_cuda11.7_cudnn8.5.0_0
### Sort-term
```markdown
python train_trajnet.py
python test_trajnet.py
```
### Long-term
```markdown
python train_long.py
python test_long.py
```
### End to End Prediction
```markdown
python real_trajnet.py
```
## Abstract
This research focuses on developing a solution for the challenge of creating models that can predict the pedestrian’s future trajectory by only observing some past trajectories. We propose an autoencoder-based model, combine with three different modules made up of the CoordConv to enhance the coordinate features. The modules namely past trajectory encoder, endpoint decoder, and future trajectory decoder, generate the future trajectory maps by input the past trajectory and the scene information map, due to predict the future is a multi-modality problem, we predict the possible endpoints of the target, then generate the trajectory from the start predict position to each endpoint. With the order of predicting the endpoints first and then the trajectory, and inputting the scene information map, our prediction makes a more reasonable prediction, and so on we have achieved good results in endpoint prediction and trajectory prediction. In addition, with the use of a CNN-based model, despite the future predict step being fixed, our model can predict multiple steps in one prediction, this can speed up the prediction process, therefore our proposed model can be predicted the trajectory in real time.
![image](https://github.com/ycchen218/TrajCoordAE/blob/master/git_image/traj.png)
## Model Overview
![image](https://github.com/ycchen218/TrajCoordAE/blob/master/git_image/model_overview.png)
## End-to-End Real-time Result
YOLOv5+DeepSORT+CoordAutoencoder
![image](https://github.com/ycchen218/TrajCoordAE/blob/master/git_image/movie.gif)

## Reference
1. [Ynet](https://github.com/HarshayuGirase/Human-Path-Prediction)<br>
2. [YOLOv5+DeepSORT](https://github.com/HowieMa/DeepSORT_YOLOv5_Pytorch)
