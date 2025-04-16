import os
import numpy as np
import json
import cv2
import open3d as o3d
import pyquaternion as pyquat
from reverse_projection import image_reverse_projection, point_reverse_projection
#copy from movi_f README
"""python
if "forward_flow" in layers:
    result["metadata"]["forward_flow_range"] = [
        data_ranges["forward_flow"]["min"] / scale * 512,
        data_ranges["forward_flow"]["max"] / scale * 512]
    result["forward_flow"] = [
        subsample_nearest_neighbor(read_png(frame_path)[..., :2], 
                                   target_size)
        for frame_path in paths["forward_flow"]]
"""
# I don`t know how to express what I feel after check the details of the code
# Firstly, after check the data distribution, the image store u, v optical flow on y,z channel
# Not x, y channel as movi_f write.
# So we need to select [...,1:] from png, rather than [..., :2]
# Later, the magic number 512 don`t have any reason, we can just skip it.
# Then we get the correct results.


"""
- "forward_flow": (s, h, w, 2) [uint16]
  Forward optical flow in the form (delta_row, delta_column).
  The values are stored as uint16 and span the range specified in
  sample["metadata"]["forward_flow_range"]. To convert them back to pixels use:
    minv, maxv = sample["metadata"]["forward_flow_range"]
    depth = sample["forward_flow"] / 65535 * (maxv - minv) + minv
"""

def read_forward_flow(img_path, data_ranges_path):
    """
    Reads and processes the forward flow data from an image file and a JSON file containing data ranges.
    Args:
        img_path (str): The file path to the image containing the forward flow data.
        data_ranges_path (str): The file path to the JSON file containing the data ranges for normalization.
    Returns:
        numpy.ndarray: A 2D array representing the processed forward flow data with normalized values.
    Notes:
        - this is different from the original movi_f code, check comments for detail.
        - The function reads the forward flow image and selects only the y and z channels (ignoring the x channel).
        - The forward flow values are normalized using the min and max values provided in the JSON file.
        - The normalization formula used is:
          `normalized_value = (raw_value / 65535) * (max_value - min_value) + min_value`
    """
    with open(data_ranges_path, 'r') as f:
        data_ranges = json.load(f)
    forward_flow_max = data_ranges["forward_flow"]["max"]
    forward_flow_min = data_ranges["forward_flow"]["min"]
    minv, maxv = forward_flow_min, forward_flow_max # skip multi 512

    forward_flow = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    forward_flow = forward_flow[...,1:] # select y,z channel, not x,y channel
    forward_flow = forward_flow / 65535 * (maxv - minv) + minv

    return forward_flow

def read_segmentation(img_path):
    """
    Reads and processes the segmentation data.
    Args:
        img_path (str): The file path to the image containing the segmentation data.
    Returns:
        numpy.ndarray: A 2D array representing the processed segmentation data.
    """
    segmentation = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return segmentation


def clamp(value,min_val,max_val):
    """
    Clamp a given value within a specified range.
    Parameters:
    -----------
    value : float or int
        The value to be clamped.
    min_val : float or int
        The minimum allowable value.
    max_val : float or int
        The maximum allowable value.
    Returns:
    --------
    float or int
        The clamped value, which will be:
        - `min_val` if `value` is less than `min_val`.
        - `max_val` if `value` is greater than `max_val`.
        - `value` if it is within the range [min_val, max_val].
    """
    return min(max(value,min_val),max_val)




def camera_space_to_world_space(camera_space_points, camera_position, camera_quaternion):
    """
    Convert camera space points to world space points using camera position and quaternion.
    Args:
        camera_space_points (numpy.ndarray): 3D points in camera space.
        camera_position (numpy.ndarray): Camera position in world space.
        camera_quaternion (numpy.ndarray): Camera orientation as a quaternion.
    Returns:
        numpy.ndarray: 3D points in world space.
    """
    rot = pyquat.Quaternion(camera_quaternion).rotation_matrix
    return (rot @ camera_space_points.T).T + camera_position.T



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
