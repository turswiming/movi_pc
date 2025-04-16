import os
import numpy as np
import json
import cv2
import open3d as o3d
import pyquaternion as pyquat
from reverse_projection import image_reverse_projection, point_reverse_projection
from util import clamp, rgb_array_to_int32
from util import read_forward_flow
from util import visualize_scene_flow

def visualize_scene_flow(camera_space_points, camera_space_points_next, scene_flow):
    """
    Function to visualize the scene flow between two sets of 3D points
    Inputs:
    - camera_space_points: 3D points in the camera space at time t
    - camera_space_points_next: 3D points in the camera space at time t+1
    - scene_flow: 3D vectors representing the flow from points at time t to t+1
    Outputs:
    - Visualizes the scene flow using Open3D, showing start points, end points, and flow vectors
    """

    # Create a point cloud for the starting points
    pcd_start = o3d.geometry.PointCloud()
    pcd_start.points = o3d.utility.Vector3dVector(camera_space_points)
    pcd_start.paint_uniform_color([0, 1, 0])  # Green color for the starting points
    pcd_end = o3d.geometry.PointCloud()
    pcd_end.points = o3d.utility.Vector3dVector(camera_space_points_next)
    pcd_end.paint_uniform_color([0, 0, 1])  # Blue color for the end points
    # Create lines to represent the scene flow vectors
    lines = []
    colors_blue = []
    colors_green = []
    for i, start_point in enumerate(camera_space_points):
        end_point = start_point + scene_flow[i]
        lines.append([i, len(camera_space_points) + i])  # Line from start to end
        colors_blue.append([0, 0, 1])  # Red color for the flow vectors
        colors_green.append([0, 1, 0])  # Green color for the flow vectors

    # Create a point cloud for the end points
    end_points = camera_space_points + scene_flow
    # pcd_end = o3d.geometry.PointCloud()
    # pcd_end.points = o3d.utility.Vector3dVector(end_points)

    # Combine start and end points
    combined_points = np.vstack((camera_space_points, end_points))
    start_half = np.vstack((camera_space_points, (camera_space_points+end_points)/2))
    end_half = np.vstack(((camera_space_points+end_points)/2,end_points))
    # Create a LineSet for the flow vectors
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(start_half)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors_green)
    line_set_end = o3d.geometry.LineSet()
    line_set_end.points = o3d.utility.Vector3dVector(end_half)
    line_set_end.lines = o3d.utility.Vector2iVector(lines)
    line_set_end.colors = o3d.utility.Vector3dVector(colors_blue)
    # Create a LineSet for the flow vectors
    
    # Visualize the scene flow
    o3d.visualization.draw_geometries([pcd_start, line_set,line_set_end,pcd_end], window_name="Scene Flow Visualization")




def process_one_dataset(dataset_path,show_axis=False,visualize=False):
    metadata_path = os.path.join(dataset_path, "metadata.json")
    data_range_path = os.path.join(dataset_path, "data_ranges.json")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    K = np.array(metadata["camera"]["K"]).reshape(3, 3)
    positions = metadata["camera"]["positions"]
    quaternions = metadata["camera"]["quaternions"]

    pcs = []
    i = 0
    camera_space_points_list = []
    forward_flow_list = []
    camera_position_list = []
    rotation_list = []
    for file in sorted(os.listdir(dataset_path)):

        if file.endswith(".tiff") and file.startswith("depth"):
            camera_position = positions[i]
            camera_quaternion = quaternions[i]
            i += 1
            camera_position = np.array(camera_position).reshape(3, 1)
            camera_quaternion = np.array(camera_quaternion).reshape(4, 1)
            camera_position_list.append(camera_position)
            camera_quaternion = np.array(camera_quaternion).flatten()
            camera_quaternion = camera_quaternion / np.linalg.norm(camera_quaternion)

            rot = pyquat.Quaternion(camera_quaternion).rotation_matrix
            rotation_list.append(rot)
            dep_img_path = os.path.join(dataset_path, file)
            distance = cv2.imread(dep_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            forward_flow_path = dep_img_path.replace("depth", "forward_flow").replace(".tiff", ".png")
            res = metadata["flags"]["resolution"]
            internal_matrix = K * res
            fx, fy = internal_matrix[0, 0], internal_matrix[1, 1]
            cx, cy = internal_matrix[0, 2], internal_matrix[1, 2]
            camera_space_points = image_reverse_projection(distance, fx, fy, cx, cy)
            camera_space_points_list.append(camera_space_points)
            forward_flow = read_forward_flow(forward_flow_path, data_range_path)
            forward_flow_list.append(forward_flow)
            # global_space_points = (rot @ camera_space_points.T).T + camera_position.T


    for i in range(len(camera_space_points_list)):
        forward_flow = forward_flow_list[i]
        camera_space_points = camera_space_points_list[i]
        camera_space_points_next = camera_space_points_list[i+1]
        rot = rotation_list[i]
        rot_next = rotation_list[i+1]
        camera_position = camera_position_list[i]
        camera_position_next = camera_position_list[i+1]
        v_mesh, u_mesh = np.meshgrid(np.arange(distance.shape[1]), np.arange(distance.shape[0]))
        mesh = np.stack((u_mesh, v_mesh), axis=-1)
        forward_flow = forward_flow.reshape(-1, 2)
        #replace x and y
        mesh = mesh.reshape(-1, 2)
        forward_flow_next = mesh.reshape(-1, 2) + forward_flow[...,[1,0]]
        camera_space_points = camera_space_points.reshape(-1, 3)
        camera_space_points_next = camera_space_points_next.reshape(-1, 3)
        global_space_points = (rot @ camera_space_points.T).T + camera_position.T
        global_space_points_next = (rot_next @ camera_space_points_next.T).T + camera_position_next.T
        global_space_points = global_space_points.reshape(res, res, 3)
        global_space_points_next = global_space_points_next.reshape(res, res, 3)
        global_space_points_next_place = [global_space_points_next[clamp(int(u),0,res-1),clamp(int(v),0,res-1)] for u,v in forward_flow_next]
        global_space_points_next_place = np.asarray(global_space_points_next_place)
        scene_flow = global_space_points_next_place-global_space_points.reshape(-1,3)
        global_space_points = global_space_points.reshape(-1, 3)
        global_space_points_next = global_space_points_next.reshape(-1, 3)
        if visualize:
            # Visualize the scene flow
            visualize_scene_flow(global_space_points, global_space_points_next, scene_flow)

        #visualize
    if len(pcs) > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        pcs.append(axis)
        o3d.visualization.draw_geometries(pcs, window_name="Point Cloud", width=800, height=600)





path = "/home/lzq/workspace/movi-f/outputs/"

for dataset in sorted(os.listdir(path),key=lambda x: int(x)):
    process_one_dataset(os.path.join(path, dataset),show_axis=False,visualize=True)
