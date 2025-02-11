import numpy as np
import time
import genesis as gs

import trimesh
import numpy as np
from PIL import Image
import os
import argparse
import cv2
import open3d as o3d
import copy
#import torch
import plyfile

# import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv, det
# from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
from scipy.spatial.transform import Rotation
import transforms3d as t3d

import sys
import warnings
import os
import yaml
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_dir, '../'))
from utils import *
from math_tools import *

from pathlib import Path

def main():


    parser = argparse.ArgumentParser(description="data processing the traj.")
    parser.add_argument("-d", "--data_index", type=int, default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="hang_mug",  help="Input task name.")
    parser.add_argument("-debug", "--debug", default=0,  help="project name.")     
    args = parser.parse_args()

    env = load_yaml( "/home/jiahe/data/config/env.yaml" )
    world_2_head = np.array( env.get("world_2_head") )
    world_2_left_base = np.array( env.get("world_2_left_base") )
    world_2_right_base = np.array( env.get("world_2_right_base") )
    left_ee_2_left_cam = np.array( env.get("left_ee_2_left_cam") )    
    right_ee_2_right_cam = np.array( env.get("right_ee_2_right_cam") )
    left_bias = world_2_left_base
    left_tip_bias = np.array( env.get("left_ee_2_left_tip") )
    right_bias = world_2_right_base
    right_tip_bias = np.array( env.get("right_ee_2_right_tip") )
    head_bound_box = np.array( env.get("head_bounding_box") )
    hand_bound_box = np.array( env.get("hand_bounding_box") )
    o3d_data = env.get("intrinsic_o3d")[0]
    cam_intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(o3d_data[0], o3d_data[1], o3d_data[2], o3d_data[3], o3d_data[4], o3d_data[5])


    task_name = args.task 
    print("task_name: ", task_name)
    task_config = load_yaml( "/home/jiahe/data/real2sim/{}/task.yaml".format(task_name) )

    obj0_name = task_config.get("objects")[0]
    obj1_name = task_config.get("objects")[1]


    ep_data = np.load( "/home/jiahe/data/real2sim/{}/traj/{}.npy".format(task_name, args.data_index), allow_pickle = True)
    # ep_data = ep_data.item()
    # print("ep_data: ", ep_data)
    obj0_pcd_file = "/home/jiahe/data/real2sim/object_dino/{}.npy".format(obj0_name)
    obj1_pcd_file = "/home/jiahe/data/real2sim/object_dino/{}.npy".format(obj1_name)
    obj0_featured_pcd = np.load(obj0_pcd_file, allow_pickle=True)
    obj1_featured_pcd = np.load(obj1_pcd_file, allow_pickle=True)

    obj0_config = load_yaml( "/home/jiahe/data/real2sim/{}/first_frame/{}_0.yaml".format(task_name, args.data_index) )
    obj0_pose = np.array( obj0_config.get("pose") )
    obj0_center = np.array( obj0_config.get("center") )
    obj0_scale = np.array( obj0_config.get("scale") )

    obj1_config = load_yaml( "/home/jiahe/data/real2sim/{}/first_frame/{}_1.yaml".format(task_name, args.data_index) )
    obj1_pose = np.array( obj1_config.get("pose") )
    obj1_center = np.array( obj1_config.get("center") )
    obj1_scale = np.array( obj1_config.get("scale") )

    cam_extrinsic = world_2_head

    traj_data = np.load( "/home/jiahe/data/real2sim/processed_traj/{}/ep{}.npy".format(task_name, args.data_index), allow_pickle = True)
    print("len: ", len(traj_data))
    img_size = 512
    fxfy = float(img_size)

    intrinsic_np = np.array([
        [fxfy, 0., img_size/2],
        [0. ,fxfy, img_size/2],
        [0., 0., 1.0]
    ])


    obj0_pose=np.array(  [[-0.42532016, -0.0692748,   0.90238781,  0.33756204],
        [ 0.90448226,  0.00255653,  0.42650359,  0.11597637],
        [-0.03185293,  0.99759434,  0.06157049,  0.14509892],
        [ 0.,          0.,          0.,          1.        ]])
    obj1_pose= np.array(  [[ 0.97238062, -0.02779443, -0.23173994,  0.33754128],
        [-0.23094084,  0.02919831, -0.97252958, -0.17120192],
        [ 0.03379732,  0.99918713,  0.021973,    0.00821233],
        [ 0.,          0.,          0.,          1.        ]])


    obj0_points = obj0_featured_pcd[:, 0:3] #- obj0_center  # xyz + dinofeature
    obj0_pcd = numpy_2_pcd( obj0_points * obj0_scale, np.array( [0., 1., 0.] ))
    obj0_pcd = obj0_pcd.transform(obj0_pose)

    ########################## init ##########################
    gs.init(backend=gs.gpu, precision="32")
    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.0001,
            # damp=
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
        ),
        show_viewer=True,
        # show_viewer=False,
    )

    ########################## entities ##########################
    # plane = scene.add_entity(
    #     gs.morphs.Plane(),
    # )
    
    obj0_pos = obj0_pose[0:3,3]
    print("obj0_center: ", obj0_center)
    obj0_rot = Rotation.from_matrix( obj0_pose[0:3,0:3] )
    obj0_quat = obj0_rot.as_quat()
    obj0_quat = ( obj0_quat[3], obj0_quat[0], obj0_quat[1], obj0_quat[2])
    obj0 = scene.add_entity(
        gs.morphs.URDF(
            file="/home/jiahe/data/urdf/{}/object.urdf".format(obj0_name),
            pos = obj0_pos,
            quat = obj0_quat,
            scale = obj0_scale,
            convexify = False,
            fixed = True
        ),
    )

    obj1_pos = obj1_pose[0:3,3]
    obj1_rot = Rotation.from_matrix( obj1_pose[0:3,0:3] )
    obj1_quat = obj1_rot.as_quat()
    obj1_quat = ( obj1_quat[3], obj1_quat[0], obj1_quat[1], obj1_quat[2])
    obj1 = scene.add_entity(
        gs.morphs.URDF(
            file="/home/jiahe/data/urdf/{}/object.urdf".format(obj1_name),
            pos = obj1_pos,
            quat = obj1_quat,
            scale = obj1_scale,
            convexify = False,
            # fixed =True
        ),
    )
    ########################## build ##########################
    scene.build()


    print("obj0: ", obj0.get_pos())    
    # print("obj1: ", obj1.get_pos())
    for i in range(len(traj_data)):
        for j in range(10):
            obj1_pos = traj_data[-1][0:3,3]
            obj1_rot = Rotation.from_matrix( traj_data[i][0:3,0:3] )
            obj1_quat = obj1_rot.as_quat()
            obj1_quat = ( obj1_quat[3], obj1_quat[0], obj1_quat[1], obj1_quat[2])

            obj1.set_pos(obj1_pos)
            obj1.set_quat(obj1_quat)
            scene.step()

    for i in range(400):
        scene.step()

if __name__ == "__main__":
    main()