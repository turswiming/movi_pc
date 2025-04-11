# this repo aims to reconstruct point cloud from depth images
# and camera intrinsics

import os
import numpy as np
import json
import cv2
import open3d as o3d
import math
import pyquaternion as pyquat
import numpy as np

def look_at_rotation(position):
    position = np.array(position, dtype=np.float64)
    if np.linalg.norm(position) < 1e-6:
        return np.eye(3)  # 物体已在原点，返回单位矩阵
    
    # 计算前向向量（指向原点）
    forward = -position / np.linalg.norm(position)
    
    # 临时上向量（默认Y轴）
    up_temp = np.array([0, 1, 0], dtype=np.float64)
    
    # 计算右向量
    right = np.cross(up_temp, forward)
    if np.linalg.norm(right) < 1e-6:
        # 处理前向与Y轴平行的情况，改用Z轴
        up_temp = np.array([0, 0, 1], dtype=np.float64)
        right = np.cross(up_temp, forward)
    right /= np.linalg.norm(right)
    
    # 计算实际上向量
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    
    # 构造旋转矩阵（列向量为右、上、前向）
    rotation_matrix = np.column_stack((right, up, forward))
    return rotation_matrix

def load_depth_image(depth_path):
    # 加载元数据
    json_path = os.path.join(depth_path, "metadata.json")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    K = np.array(metadata["camera"]["K"]).reshape(3, 3)
    R = np.array(metadata["camera"]["R"]).reshape(4, 4)
    field_of_view = metadata["camera"]["field_of_view"] # rad metrics
    focal_length = metadata["camera"]["focal_length"]
    sensor_width = metadata["camera"]["sensor_width"]
    positions = metadata["camera"]["positions"]
    quaternions = metadata["camera"]["quaternions"]
    pcs = []
    i = 0
    for file in sorted(os.listdir(depth_path)):

        if file.endswith(".tiff") and file.startswith("depth"):
            camera_position = positions[i]
            camera_quaternion = quaternions[i]
            camera_position = np.array(camera_position).reshape(3, 1)
            camera_quaternion = np.array(camera_quaternion).reshape(4, 1)
            # 计算旋转矩阵
            camera_quaternion = np.array(camera_quaternion).flatten()
            camera_quaternion = camera_quaternion / np.linalg.norm(camera_quaternion)

            rot = pyquat.Quaternion(camera_quaternion).rotation_matrix
            # 计算平移矩阵
            translation_matrix = np.array([
                [1, 0, 0, camera_position[0][0]],
                [0, 1, 0, camera_position[1][0]],
                [0, 0, 1, camera_position[2][0]],
                [0, 0, 0, 1]
            ], dtype=np.float64)

            i += 1
            img_path = os.path.join(depth_path, file)
            distance = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            u_mesh_origin, v_mesh_origin = np.meshgrid(np.arange(distance.shape[1]), np.arange(distance.shape[0]))
            u_mesh_origin = u_mesh_origin.astype(np.float32)
            v_mesh_origin = v_mesh_origin.astype(np.float32)
            fx, fy = K[0, 0], K[1, 1]  # 焦距
            cx, cy = K[0, 2], K[1, 2]  # 主点偏移
            u_mesh = (u_mesh_origin/distance.shape[1] + cx) / fx *2
            v_mesh = (v_mesh_origin/distance.shape[0] + cy) / -fy *2
            angle_to_center_x = np.arctan((u_mesh) * np.tan(field_of_view / 2))
            angle_to_center_y = np.arctan((v_mesh) * np.tan(field_of_view / 2))
            x = distance * np.tan(angle_to_center_x)
            y = distance * np.tan(angle_to_center_y)
            z = distance / np.sqrt(1 + np.tan(angle_to_center_x)**2 + np.tan(angle_to_center_y)**2)
            # print("min z2-z:", np.min(z2-z))
            # print("max z2-z:", np.max(z2-z))
            camera_positions = np.stack((x, y, z), axis=-1)
            camera_positions = camera_positions.reshape(-1, 3)
            #add camera position into camera positions
            print("camera positions shape:", camera_positions.shape)
            camera_point = np.zeros((1, 3))
            print("camera point shape:", camera_point.shape)
            z_direction =  np.array([0,0,1])            #lerp from zero to z_direction
            axis = []
            for s in range(100):
                axis.append(np.array([0,0,0])*(1-s/100)+z_direction*(s/100))
            axis = np.array(axis)
            axis = axis.reshape(-1, 3)
            camera_positions = np.concatenate((camera_positions, axis), axis=0)
            # camera_positions = axis
            project_coordinates = np.stack((u_mesh_origin, v_mesh_origin, distance), axis=-1)
            camera_positions -= camera_position.T
            global_positions = camera_positions @ rot
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(camera_positions)
            #save point cloud
            path = os.path.join(depth_path, "pointcloud_camera"+file.split("depth")[-1].split(".tiff")[0]+".ply")
            o3d.io.write_point_cloud(path, pcd)
            print(f"Saved point cloud to {path}")
            pcd_world = o3d.geometry.PointCloud()
            pcd_world.points = o3d.utility.Vector3dVector(global_positions.reshape(-1, 3))
            path = os.path.join(depth_path, "pointcloud_world"+file.split("depth")[-1].split(".tiff")[0]+".ply")
            o3d.io.write_point_cloud(path, pcd_world)
            pcd_world.paint_uniform_color([i/24, 0, 0])
            print(f"Saved point cloud to {path}")
            # pcs.append(pcd)
            pcs.append(pcd_world)
        #visualize
    if len(pcs) > 0:
        o3d.visualization.draw_geometries(pcs, window_name="Point Cloud", width=800, height=600)
        return





path = "/home/lzq/workspace/movi-f/outputs/"

for dataset in sorted(os.listdir(path),key=lambda x: int(x)):
    load_depth_image(os.path.join(path, dataset))
