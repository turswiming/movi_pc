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

x_y_z_scale_factor = [1,1,-1]

def reverse_projection(u,v,fx,fy,cx,cy,distance_map)-> tuple[float, float, float]:
    """
    Parameters:
        u (float): Pixel x-coordinate.
        v (float): Pixel y-coordinate.
        fx (float): Focal length in the x direction.
        fy (float): Focal length in the y direction.
        cx (float): Principal point x-coordinate (negative value, same as in movi_f).
        cy (float): Principal point y-coordinate (negative value, same as in movi_f).
        distance_map (np.ndarray): Distance map (originated from movi_f, not a depth map).

    Returns:
        tuple[float, float, float]: 3D point in camera space (x, y, z).
    """
    angle_to_center_x = (u + cx) / fx 
    angle_to_center_y = (v + cy) / fy

    distance = distance_map[int(v), int(u)]
    x = 1 * angle_to_center_x
    y = 1 * angle_to_center_y
    z = 1 / np.sqrt(1 + x**2 + y**2)
    x = distance * x
    y = distance * y
    z = distance * z
    return x, y, z
    

def load_depth_image(depth_path,show_axis=False,visualize=False):
    json_path = os.path.join(depth_path, "metadata.json")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    K = np.array(metadata["camera"]["K"]).reshape(3, 3)
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
            i += 1
            camera_position = np.array(camera_position).reshape(3, 1)
            camera_quaternion = np.array(camera_quaternion).reshape(4, 1)

            camera_quaternion = np.array(camera_quaternion).flatten()
            camera_quaternion = camera_quaternion / np.linalg.norm(camera_quaternion)

            rot = pyquat.Quaternion(camera_quaternion).rotation_matrix

            dep_img_path = os.path.join(depth_path, file)
            distance = cv2.imread(dep_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            rgba_img_path = dep_img_path.replace("depth", "rgba").replace(".tiff", ".png")
            rgba_img = cv2.imread(rgba_img_path, cv2.IMREAD_UNCHANGED)
            rgba_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2RGB)
            rgba_img = rgba_img.astype(np.float32)
            rgba_img = rgba_img / 255.0
            rgba_img = rgba_img.reshape(-1, 3)
            u_mesh_origin, v_mesh_origin = np.meshgrid(np.arange(distance.shape[1]), np.arange(distance.shape[0]))
            u_mesh_origin = u_mesh_origin.astype(np.float32)
            v_mesh_origin = v_mesh_origin.astype(np.float32)
            internal_matrix = K * metadata["flags"]["resolution"]
            fx, fy = internal_matrix[0, 0], internal_matrix[1, 1]
            cx, cy = internal_matrix[0, 2], internal_matrix[1, 2]
            u_mesh = (u_mesh_origin + cx) / fx
            v_mesh = (v_mesh_origin + cy) / fy

            z = distance / np.sqrt(1 + u_mesh**2 + v_mesh**2)
            x = z * u_mesh
            y = z * v_mesh
            camera_space_points = np.stack((x_y_z_scale_factor[0]*x, x_y_z_scale_factor[1]*y, x_y_z_scale_factor[2]*z), axis=-1)# i test this...
            
            camera_space_points = camera_space_points.reshape(-1, 3)
            if show_axis:
                #add camera position into camera positions
                z_direction =  np.array([0,0,1])            #lerp from zero to z_direction
                axis = []
                axis_color = []
                for s in range(100):
                    axis.append(np.array([0,0,0])*(1-s/100)+z_direction*(s/100))
                    axis_color.append([0,0,1])
                axis = np.array(axis)
                axis = axis.reshape(-1, 3)
                camera_space_points = np.concatenate((camera_space_points, axis), axis=0)
                rgba_img = np.concatenate((rgba_img, axis_color), axis=0)
            
            global_space_points = (rot @ camera_space_points.T).T + camera_position.T


            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(camera_space_points)
            pcd.colors = o3d.utility.Vector3dVector(rgba_img)
            #save point cloud
            path = os.path.join(depth_path, "pointcloud_camera"+file.split("depth")[-1].split(".tiff")[0]+".ply")
            if o3d.io.write_point_cloud(path, pcd):
                print(f"Saved point cloud to {path}")
            pcd_world = o3d.geometry.PointCloud()
            pcd_world.points = o3d.utility.Vector3dVector(global_space_points)
            pcd_world.colors = o3d.utility.Vector3dVector(rgba_img)
            path = os.path.join(depth_path, "pointcloud_world"+file.split("depth")[-1].split(".tiff")[0]+".ply")
            if o3d.io.write_point_cloud(path, pcd_world):
                print(f"Saved point cloud to {path}")

            # pcd.paint_uniform_color([i/24, 0, 0])
            # pcd_world.paint_uniform_color([0, 0, i/24])
            # pcs.append(pcd)
            if visualize:
                pcs.append(pcd_world)
        #visualize
    if len(pcs) > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        pcs.append(axis)
        o3d.visualization.draw_geometries(pcs, window_name="Point Cloud", width=800, height=600)





path = "/home/lzq/workspace/movi-f/outputs/"

for dataset in sorted(os.listdir(path),key=lambda x: int(x)):
    load_depth_image(os.path.join(path, dataset),show_axis=False,visualize=False)
