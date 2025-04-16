import os
import numpy as np
import json
import cv2
import open3d as o3d
import pyquaternion as pyquat
from reverse_projection import image_reverse_projection, point_reverse_projection
from util import clamp, rgb_array_to_int32, camera_space_to_world_space
from util import read_forward_flow,read_segmentation
from util import visualize_scene_flow, vis,visualize_point_trajectory





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

def get_obj_rotation_list_list(metadata):
    """
    Get the object rotation list from the metadata.
    Args:
        metadata (dict): Metadata containing object rotations.
    Returns:
        list: List of object rotation matrices.
    """
    object_rotation_list_list = []
    for instance in metadata["instances"]:
        object_rotation_list = []
        for j in range(len(instance["quaternions"])):
            quaternions = np.array(instance["quaternions"][j])
            rot = pyquat.Quaternion(quaternions).rotation_matrix
            object_rotation_list.append(rot)
        object_rotation_list_list.append(object_rotation_list)
    return object_rotation_list_list

def get_obj_center_list(metadata):
    """
    Get the object center list from the metadata.
    Args:
        metadata (dict): Metadata containing object centers.
    Returns:
        list: List of object centers.
    """
    object_center_list_list = []
    for instance in metadata["instances"]:
        object_center_list = []
        for j in range(len(instance["positions"])):
            position = instance["positions"][j]
            object_center_list.append(position)
        object_center_list_list.append(object_center_list)
    return object_center_list_list  

def get_metadata(dataset_path):
    """
    Get the metadata from the dataset path.
    Args:
        dataset_path (str): Path to the dataset.
    Returns:
        dict: Metadata containing camera parameters and object instances.
    """
    metadata_path = os.path.join(dataset_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def get_object_rotation_tensors(res,num_frames,num_objects, indices_of_instance,object_rotation_list_list):
    """
    Get the object rotation tensors for each frame.
    Args:
        res (int): Resolution of the images.
        num_frames (int): Number of frames in the dataset.
        num_objects (int): Number of objects in the dataset.
        indices_of_instance (numpy.ndarray): Indices of instances in the segmentation.
        object_rotation_list_list (list): List of object rotation matrices.
    Returns:
        list: List of object rotation tensors for each frame.
    """
    object_rotation_tensor = np.zeros((res,res,3,3), dtype=np.float32)
    #object rotation tensor is a 4D tensor, the last dimension is 3x3 matrix
    #defult is eye(3)
    object_rotation_tensor[...,0,0] = 1
    object_rotation_tensor[...,1,1] = 1
    object_rotation_tensor[...,2,2] = 1
    object_rotation_tensor_list = []
    for i in range(num_frames):
        object_rotation_tensor_list.append(np.copy(object_rotation_tensor))
    for i in range(num_frames):
        for j in range(num_objects):
            object_rotation_tensor_list[i][indices_of_instance == j] = object_rotation_list_list[j][i]
    return object_rotation_tensor_list

def process_one_dataset(dataset_path,show_axis=False,visualize=False):
    metadata = get_metadata(dataset_path)

    K = np.array(metadata["camera"]["K"]).reshape(3, 3)
    positions = metadata["camera"]["positions"]
    quaternions = metadata["camera"]["quaternions"]
    res = metadata["flags"]["resolution"]
    num_frames = len(positions)
    global_space_points_list = []
    segmentation_list = []
    seg_color_to_instance_ID_map = {}
    for i in range(num_frames):

        camera_position = positions[i]
        camera_quaternion = quaternions[i]
        camera_position = np.array(camera_position).reshape(3, 1)
        camera_quaternion = np.array(camera_quaternion).reshape(4, 1)
        camera_quaternion = camera_quaternion.flatten()

        dep_img_path = os.path.join(dataset_path, f"depth_{i:05d}.tiff")
        distance = cv2.imread(dep_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        segmentation_path = os.path.join(dataset_path, f"segmentation_{i:05d}.png")
        segmentation = read_segmentation(segmentation_path)
        
        internal_matrix = K * res
        fx, fy = internal_matrix[0, 0], internal_matrix[1, 1]
        cx, cy = internal_matrix[0, 2], internal_matrix[1, 2]
        camera_space_points = image_reverse_projection(distance, fx, fy, cx, cy)
        camera_space_points = camera_space_points.reshape(-1, 3)
        global_space_points = camera_space_to_world_space(camera_space_points, camera_position, camera_quaternion)
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
    num_objects = len(seg_color_to_instance_ID_map)
    
    object_rotation_list_list = get_obj_rotation_list_list(metadata)
    global_space_center_list_list = get_obj_center_list(metadata)
    
    for frame in range(num_frames):
        segmentation = segmentation_list[frame]
        global_space_points = global_space_points_list[frame]

        indices_of_instance = np.zeros_like(segmentation, dtype=np.int8)
        indices_of_instance -= 1 #set background to -1 so that in following processing we can ignore it
        #set the instance id to the segmentation
        for color, instance_id in seg_color_to_instance_ID_map.items():
            indices_of_instance[segmentation == color] = instance_id
        
        global_space_center_tensor = np.zeros((res,res,3), dtype=np.float32)
        global_space_center_tensor_list = []
        for i in range(num_frames):
            global_space_center_tensor_list.append(np.copy(global_space_center_tensor))
        for i in range(num_frames):
            for j in range(num_objects):
                global_space_center_tensor_list[i][indices_of_instance == j] = global_space_center_list_list[j][i]
        global_space_center_tensor = global_space_center_tensor_list[frame]
        global_space_center_tensor = global_space_center_tensor.reshape(-1, 3)


        object_rotation_tensor_list = get_object_rotation_tensors(res, num_frames, num_objects, indices_of_instance, object_rotation_list_list)
        object_rotation_tensor = object_rotation_tensor_list[frame]
        object_rotation_tensor = object_rotation_tensor.reshape(-1, 3, 3)
        
        object_space_points =  np.einsum('nji,nj->ni', object_rotation_tensor, (global_space_points - global_space_center_tensor))
        #compute world space points num_frames
        global_space_trajectories_list = []
        for i in range(num_frames):
            object_rotation_tensor = object_rotation_tensor_list[i]
            object_rotation_tensor = object_rotation_tensor.reshape(-1, 3, 3)
            global_space_trajectory = np.einsum('nij,nj->ni', object_rotation_tensor, (object_space_points))
            global_space_trajectory = global_space_trajectory + global_space_center_tensor_list[i].reshape(-1, 3)
            global_space_trajectories_list.append(global_space_trajectory)
            
        global_space_trajectories_tensor = np.array(global_space_trajectories_list)
        global_space_trajectories_tensor = global_space_trajectories_tensor.astype(np.float16) #to save space
        traj_save_path = os.path.join(dataset_path, f"global_space_trajectories_{frame:05d}.npz")
        try:
            np.savez_compressed(traj_save_path, global_space_trajectories_tensor)
        except Exception as e:
            print(f"Error saving file: {e}")
        else:
            print(f"Saved successfully: {traj_save_path}")
            print(f"file size: {os.path.getsize(traj_save_path) / 1024 / 1024} MB")
        if visualize:
            # visualize_scene_flow(global_space_points, global_space_points_next, scene_flow)
            visualize_point_trajectory(global_space_trajectories_tensor)





path = "/home/lzq/workspace/movi-f/outputs/"

for dataset in sorted(os.listdir(path),key=lambda x: int(x)):
    process_one_dataset(os.path.join(path, dataset),show_axis=False,visualize=False)
