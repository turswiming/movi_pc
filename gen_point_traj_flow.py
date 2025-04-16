import os
import numpy as np
import json
import cv2
import open3d as o3d
import pyquaternion as pyquat
from reverse_projection import image_reverse_projection, point_reverse_projection
from util import clamp, rgb_array_to_int32, camera_space_to_world_space
from util import read_forward_flow,read_segmentation
from util import visualize_scene_flow





def most_likely_instance(segmentations,color,instances)-> int:
    """
    Identify the most likely color for a given segmentation.
    Args:
        segmentation (numpy.ndarray): Segmentation image.
        instances (list): List of instances object, same in metadata.
    Returns:
        int: The most likely color.
    """
    # Count the occurrences of each instance in the segmentation
    gt_features = [instance["visibility"] for instance in instances]
    color_feature = [len(segmentation[segmentation==color]) for segmentation in segmentations]
    gt_features = np.array(gt_features)
    color_feature = np.array(color_feature)
    #compute the most likely instance
    diff = []
    for gt_feature in gt_features:
        difference = sum(abs(gt_feature - color_feature))
        diff.append(difference)    
    #get the min index
    diff = np.array(diff)
    min_index = np.argmin(diff, axis=0)
    return min_index



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
    global_space_points_list = []
    forward_flow_list = []
    camera_position_list = []
    rotation_list = []
    segmentation_list = []
    seg_color_to_instance_ID_map = {}
    for file in sorted(os.listdir(dataset_path)):

        if file.endswith(".tiff") and file.startswith("depth"):
            camera_position = positions[i]
            camera_quaternion = quaternions[i]
            i += 1
            camera_position = np.array(camera_position).reshape(3, 1)
            camera_quaternion = np.array(camera_quaternion).reshape(4, 1)
            camera_quaternion = np.array(camera_quaternion).flatten()

            dep_img_path = os.path.join(dataset_path, file)
            distance = cv2.imread(dep_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            forward_flow_path = dep_img_path.replace("depth", "forward_flow").replace(".tiff", ".png")
            forward_flow = read_forward_flow(forward_flow_path, data_range_path)

            segmentation_path = dep_img_path.replace("depth", "segmentation").replace(".tiff", ".png")
            segmentation = read_segmentation(segmentation_path)
            
            res = metadata["flags"]["resolution"]
            internal_matrix = K * res
            fx, fy = internal_matrix[0, 0], internal_matrix[1, 1]
            cx, cy = internal_matrix[0, 2], internal_matrix[1, 2]
            camera_space_points = image_reverse_projection(distance, fx, fy, cx, cy)
            camera_space_points = camera_space_points.reshape(-1, 3)
            global_space_points = camera_space_to_world_space(camera_space_points, camera_position, camera_quaternion)
            
            camera_position_list.append(camera_position)
            camera_space_points_list.append(camera_space_points)
            forward_flow_list.append(forward_flow)
            global_space_points_list.append(global_space_points)
            segmentation = rgb_array_to_int32(segmentation)
            segmentation_list.append(segmentation)
    # identify the instance ID
    #convert 8 bit r, g, b to int 32
    unique_colors = np.unique(np.array(segmentation_list))
    for color in unique_colors:
        if color == 0:
            continue
        if color not in seg_color_to_instance_ID_map:
            #return the most likely id
            id = most_likely_instance(segmentation_list,color,metadata["instances"])
            seg_color_to_instance_ID_map[color] = id
    if len(list(set(seg_color_to_instance_ID_map.values()))) != len(unique_colors)-1:
        raise Exception(
            f"segmentation color not match instance, please consider remove the dataset{dataset_path}")
    
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
            visualize_scene_flow(global_space_points, global_space_points_next, scene_flow)

        #visualize
    if len(pcs) > 0:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        pcs.append(axis)
        o3d.visualization.draw_geometries(pcs, window_name="Point Cloud", width=800, height=600)





path = "/home/lzq/workspace/movi-f/outputs/"

for dataset in sorted(os.listdir(path),key=lambda x: int(x)):
    process_one_dataset(os.path.join(path, dataset),show_axis=False,visualize=True)
