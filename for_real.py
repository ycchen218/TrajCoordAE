import torch
import torch.nn as nn
from utils.image_utils import sampling,get_patch_grad
import numpy as np
import pandas as pd
import cv2
import time
import argparse
from YOLO_utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from VideoTrack import VideoTracker
from utils_ds.draw import draw_boxes

def drawFrame(framedata,future_samples,agent_ids,resize,frame):
    pos_xs = (framedata["x"]//resize).tolist()
    pos_ys = (framedata["y"]//resize).tolist()
    future_samples = (future_samples//resize).astype(np.int32)
    for i in range(len(agent_ids)):
        cv2.putText(frame, text=str(int(agent_ids[i])), org=(int(pos_xs[i]), int(pos_ys[i])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75, color=(0, 0, 255), thickness=2)
        for future_sample in future_samples[i]:
            future_sample = future_sample.reshape((-1, 1, 2))
            cv2.polylines(frame, [future_sample], False, (0, 255, 0), 2,lineType=cv2.LINE_AA)
            cv2.drawMarker(frame, (future_sample[-1][0,0], future_sample[-1][0,1]), (255, 0, 0), markerType=2,thickness=2)
    return frame

def delData(agent_ids,dataList,a_id):
    for c,a in enumerate(a_id):
        if a not in agent_ids:
            a_id.pop(c)
            dataList.pop(c)
    return a_id,dataList

def ForModel(model,dataList,scene_image,device,waypoints,temperature,obs_len,num_goals):
    # start = time.time()
    trajectory = []
    for dl in dataList:
        x = dl.loc[:,"x"]
        y = dl.loc[:,"y"]
        xy = torch.as_tensor(np.array([x,y]).T)
        trajectory.append(xy)
    trajectory = torch.stack(trajectory)
    _, _, H, W = scene_image.shape
    observed = trajectory.to(device)
    observed_map = get_patch_grad(observed, H, W, device=device)
    observed_map = torch.stack(observed_map).reshape([-1, obs_len, H, W])
    semantic_image = scene_image.expand(observed_map.shape[0], -1, -1, -1)
    feature_input = torch.cat([semantic_image, observed_map], dim=1)
    features = model.pred_feature(feature_input)
    pred_waypoint_map = model.pred_end(features, w=W, h=H)
    pred_waypoint_map = pred_waypoint_map[:, waypoints]
    pred_waypoint_map_sigmoid = pred_waypoint_map / temperature
    pred_waypoint_map_sigmoid = model.sigmoid(pred_waypoint_map_sigmoid)

    goal_samples = sampling(pred_waypoint_map_sigmoid[:, -1:], scene=semantic_image[:, 1], num_samples=num_goals)
    goal_samples = goal_samples.permute(2, 0, 1, 3)
    waypoint_samples = goal_samples

    future_samples = []
    for waypoint in waypoint_samples:
        waypoint_map = get_patch_grad(waypoint.reshape(-1, 2).cpu().numpy(), H, W, device=device)

        waypoint_map = torch.stack(waypoint_map).reshape([-1, len(waypoints), H, W])

        waypoint_maps_downsampled = nn.AvgPool2d(kernel_size=2, stride=2)(waypoint_map)

        pred_traj_map = model.pred_traj(features, waypoint_maps_downsampled, w=W, h=H)
        pred_traj = model.softargmax(pred_traj_map)
        future_samples.append(pred_traj)
    future_samples = torch.stack(future_samples)
    future_samples = torch.transpose(future_samples,0,1)
    return future_samples.cpu()

def PredTraj(model,framedata,agent_ids,a_id,dataList,scene_image,device,waypoints,temperature,obs_len,num_goals):

    for a in agent_ids:
        if a in a_id:
            #如果重複的在dataList代表有經過補數需要放新資料並取代以下
            #不然就是這個agent活超過8個frame我們要堆疊
            index_a = a_id.index(a)
            duplicated_data = dataList[index_a].duplicated().tolist()
            if True in duplicated_data:
                for i,dup in enumerate(duplicated_data):
                    if dup==True:
                        dataList[index_a].iloc[i]=framedata[framedata["trackId"]==a]
            if True not in duplicated_data:
                dataList[index_a] = dataList[index_a].iloc[1:]
                dataList[index_a] = pd.concat([dataList[index_a],framedata[framedata["trackId"]==a]])
        if a not in a_id:
            a_id.append(a)
            dataList.append(framedata.loc[framedata[framedata["trackId"]==a].index.repeat(8)])

    future_samples = ForModel(model,dataList,scene_image,device,waypoints,temperature,obs_len,num_goals)
    future_samples = future_samples.numpy()


    return future_samples,a_id,dataList

def r_test(model, data, val_images, num_goals, obs_len, resize, device, waypoints, temperature, exp_name,input_video_path):
    model.eval()
    with torch.no_grad():
        scene = exp_name.split("_")[0]
        scene_image = val_images[scene].to(device).unsqueeze(0)
        i = 0
        a_id = []
        dataList = []
        idList = data["frame"].drop_duplicates().tolist() #[::2]取偶數
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = time.time()
        counter = 0
        print(cap.isOpened())
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                if i in idList:
                    framedata = data.loc[data["frame"] == i].drop(columns=['sceneId', 'metaId'])
                    if framedata.size != 0:
                        framedata = framedata.drop_duplicates()
                        #[trackid, frame, x, y]
                        agent_ids = framedata["trackId"].drop_duplicates().tolist()


                        future_samples,a_id, dataList = PredTraj(model,framedata, agent_ids, a_id, dataList,scene_image,
                                                                 waypoints=waypoints,obs_len=obs_len,temperature=temperature,num_goals=num_goals,device=device)

                        frame = drawFrame(framedata,future_samples, agent_ids,resize, frame)

                        a_id, dataList = delData(agent_ids, dataList, a_id)
                if i > 10:
                    frame = drawFrame(framedata, future_samples, agent_ids, resize, frame)
                cv2.putText(frame, text=f"Frame {str(i)}", org=(5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 0, 255), thickness=2)
                counter += 1
                cv2.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("frame", frame)
                # if i>=120 and i<=240:
                #     cv2.imwrite(fr"C:\Users\user\Desktop\save_image\real\{i}.png",frame)

                start_time = time.time()
                counter = 0
                time.sleep(1 / fps)
                cv2.waitKey(1)
                i += 1

            else:
                break
        cap.release()
        cv2.destroyAllWindows()

def e2e_test(model,mask,frontend_arg,temperature,waypoints,obs_len,num_goals,resize,device):

    frontend_arg = argparse.Namespace(**frontend_arg)
    frontend_arg.img_size = check_img_size(frontend_arg.img_size)
    print(frontend_arg)
    mask = torch.as_tensor(mask).type(torch.float32).unsqueeze(0).to(device)

    lock = 0
    with VideoTracker(frontend_arg) as vdo_trk:
        yolo_time, sort_time, avg_fps = [], [], []

        idx_frame = 0
        last_out = None
        a_id = []
        dataList = []
        model.eval()
        with torch.no_grad():
            while vdo_trk.vdo.grab():
                # Inference *********************************************************************
                t0 = time.time()
                _, frame = vdo_trk.vdo.retrieve()

                if idx_frame % vdo_trk.args.frame_interval == 0:
                    outputs, yt, st = vdo_trk.image_track(frame)  # (#ID, 5) x1,y1,x2,y2,id
                    last_out = outputs
                    yolo_time.append(yt)
                    sort_time.append(st)
                    print('Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))

                else:
                    outputs = last_out  # directly use prediction in last frames
                t1 = time.time()
                avg_fps.append(t1 - t0)

                # post-processing ***************************************************************
                # visualize bbox  ********************************
                if len(outputs) > 0:

                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    frame = draw_boxes(frame, bbox_xyxy, identities)  # BGR

                    # add FPS information on output video
                    text_scale = max(1, frame.shape[1] // 1600)
                    cv2.putText(frame, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)),
                                (20, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
                    if (idx_frame%10)==0:
                        lock = 1
                        a_x = (outputs[:, 0] + outputs[:, 2]) / 2
                        a_y = (outputs[:, 1] + outputs[:, 3]) / 2
                        track_id = outputs[:, 4]
                        framedata = np.array([track_id, a_x*resize, a_y*resize]).T
                        framedata = pd.DataFrame(framedata, columns=["trackId", "x", "y"])
                        agent_ids = framedata["trackId"].drop_duplicates().tolist()

                        future_samples, a_id, dataList = PredTraj(model, framedata, agent_ids, a_id, dataList, mask,
                                                                  waypoints=waypoints, obs_len=obs_len,
                                                                  temperature=temperature, num_goals=num_goals,
                                                                  device=device)
                        frame = drawFrame(framedata, future_samples, agent_ids, resize, frame)

                        a_id, dataList = delData(agent_ids, dataList, a_id)

                if lock==1:
                    for i in range(framedata.shape[0]):
                        frame = drawFrame(framedata, future_samples, agent_ids, resize, frame)
                # if idx_frame>=265 and idx_frame<300 and (idx_frame%5)==0:
                #     cv2.imwrite(fr"C:\Users\user\Desktop\save_image\real_yolo\{idx_frame}.png",frame)


                # display on window ******************************
                if vdo_trk.args.display:
                    cv2.imshow("test", frame)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        cv2.destroyAllWindows()
                        break

                idx_frame+=1
