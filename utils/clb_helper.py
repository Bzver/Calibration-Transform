import os
import shutil

import cv2

import scipy.io as sio
import numpy as np

from typing import List, Tuple, Dict, Any

def check_video_integrity(project_dir:str, num_view:int) -> List[str]:
    lost_files = []
    for i in range(num_view):
        video_filepath = os.path.join(project_dir, "Videos", f"Camera{i+1}", "0.mp4")
        if not os.path.isfile(video_filepath):
            lost_files.append(video_filepath)
        
    return lost_files

def check_calib_integrity(root_path:str, num_view:int) -> List[str]:
    lost_files = []
    calib_dir = os.path.join(root_path, "SA_calib")
    calib_toml = os.path.join(calib_dir, "calibration.toml")
    if not os.path.isfile(calib_toml):
        lost_files.append(calib_toml)
    for i in range(num_view):
        calib_mat = os.path.join(calib_dir, "Calibration", f"hires_cam{i+1}_params.mat")
        if not os.path.isfile(calib_mat):
            lost_files.append(calib_mat)

    return lost_files

def check_dannce_mat(project_dir:str) -> bool:
    for file in os.listdir(project_dir):
        if file.endswith("_dannce.mat"):
            return True
    return False

def check_project_integrity(num_view:int, root_path:str, project_dir:str) -> bool:
    lost_video = check_video_integrity(root_path, num_view)
    lost_calib = check_calib_integrity(root_path, num_view)
    lost_files = lost_video + lost_calib

    if lost_files:
        print("These files are missing in the project:")
        for file_path in lost_files:
            print(f" - {file_path}")
        return False
    
    if not check_dannce_mat(project_dir):
        print("No '_dannce.mat' file found. Attempting to set it up...")
        calib_dir = os.path.join(root_path, "SA_Calib")
        try:
            if not duplicate_calibration_files(calib_dir, project_dir):
                print("Failed to duplicate calibration files.")
                return False

            if not generate_sync_profile(num_view, project_dir):
                print("Failed to generate sync profile.")
                return False
            
            print("Successfully set up calibration and sync files.")
        except Exception as e:
            print(f"An unexpected error occurred during file setup: {e}")
            return False

    return True

####################################################################################################################

def get_cam_count(root_path:str) -> int:
    if not os.path.isdir(root_path):
        print(f"Error: Camera videos folder not found at {root_path}")
        return 0
    count = 0
    for root, dirs, files in os.walk(root_path):
        if root.split(os.sep)[-1].startswith("Camera"):
            if "0.mp4" in files:
                count += 1
    return count

def get_calib_count(root_path:str) -> int:
    if not os.path.isdir(root_path):
        print(f"Error: Calibration folder not found at {root_path}")
        return 0
    count = 0
    for files in os.listdir(root_path):
        if files.startswith("hires_cam"):
            count += 1
    return count

def get_frame_count_cv2(video_path:str) -> int:
    """Gets the total number of frames in a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count
    except Exception as e:
        print(f"An error occurred while getting frame count: {e}")
        return None
    finally:
        cap.release()

def get_eligible_video_in_folder(video_path:str) -> Tuple[List[str], int]:
    video_list = []
    acceptable_formats = [".mp4", ".avi", ".mov", ".mkv"]
    for file in os.listdir(video_path):
        if file.startswith("Camera") and any(file.endswith(ext) for ext in acceptable_formats):
            video_list.append(os.path.join(video_path, file))
        
    return video_list, len(video_list)

####################################################################################################################

def generate_sync_profile(num_view:int, project_dir:str) -> bool:
    num_cam = get_cam_count(os.path.join(project_dir, "Videos"))
    print(f"Found {num_cam} camera views.")
    num_calib = get_calib_count(os.path.join(project_dir, "Calibration"))
    print(f"Found {num_calib} calibration files.")

    if num_cam == 0:
        print(f"Error: no camera views found in {project_dir}.")
        return False
    if num_calib == 0:
        print(f"Error: no calibration files found in {project_dir}.")
        return False
    if num_cam != num_calib:
        print(f"Error: calibration ({num_calib}) and camera views ({num_cam}) do not match!")
        return False
    if num_view > num_cam:
        print(f"Error: camera views in {project_dir} is less than project designation.")
        return False
    if num_view < num_cam:
        print(f"Using only the first {num_view} cameras.")
        num_cam = num_view

    videosample = os.path.join(project_dir, "Videos", "Camera1", "0.mp4")
    print("---Getting the frame counts---")
    framecount = get_frame_count_cv2(videosample)

    camnames = [f"Camera{i+1}" for i in range(num_cam)]
    sync, params = construct_sync_and_params(num_cam, project_dir, framecount)

    camnames_array = np.array(camnames, dtype=object)
    params_array = np.array(params, dtype=object).reshape(num_cam, 1)
    sync_array = np.array(sync, dtype=object).reshape(num_cam, 1)

    mat_data_to_save = { "camnames": camnames_array, "params": params_array, "sync": sync_array }
    output_mat_filename = "sync_dannce.mat"
    output_mat_path = os.path.join(project_dir, output_mat_filename)
    sio.savemat(output_mat_path, mat_data_to_save)
    print(f"\nAll camera sync data saved to: {output_mat_path}")
    return True

def construct_sync_and_params(num_cam:int, calib_dir:str, framecount:str, cam_names:List[str]
        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    params, sync = [], []

    for i in range(num_cam):
            print(f"---Processing {cam_names[i]}---")
            data_2d = np.zeros((framecount, 4))
            data_3d = np.zeros((framecount, 6))
            data_frame = np.arange(framecount, dtype=np.float64)
            data_sampleID = np.arange(framecount, dtype=np.float64)

            cam_struct = {
                    "data_2d": data_2d,         "data_3d": data_3d,
                    "data_frame": data_frame,   "data_sampleID": data_sampleID
                }
            
            f = os.path.join(calib_dir, "Calibration", f"hires_cam{i+1}_params.mat")
            params_struct = sio.loadmat(f)

            sync.append(cam_struct)
            params.append(params_struct)
    return sync, params

####################################################################################################################

def create_output_dirs(num_view:int, project_dir:str, mode:str) -> List[str]:
    output_dir_list = []
    for i in range(num_view):
        if mode == "calibration":
            output_dir = os.path.join(project_dir, f"view{i+1}", "calibration_images")
        elif mode == "experiment":
            output_dir = os.path.join(project_dir, "Videos", f"Camera{i+1}")
        else:
            print("Invalid mode. Expected 'calibration' or 'experiment'.")
            return
        output_dir_list.append(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created inside {project_dir}.")
    return output_dir_list

def duplicate_calibration_files(calib_dir:str, project_dir:str):
    """
    Duplicates calibration files to the project directory.
    Returns True on success, False on failure.
    """
    calib_src_path = os.path.join(calib_dir, "Calibration")
    calib_dest_path = os.path.join(project_dir, "Calibration")
    expected_calib_file_path = os.path.join(calib_dest_path, "calibration.toml")

    if not os.path.exists(calib_src_path):
        print(f"Error: Calibration source directory not found at {calib_src_path}.")
        return False

    print(f"Duplicating calibration files from {calib_src_path} to {calib_dest_path}...")
    try:
        shutil.copytree(calib_src_path, calib_dest_path, dirs_exist_ok=True)
        if os.path.exists(expected_calib_file_path):
            print(f"Calibration files duplicated to {calib_dest_path} successfully.")
            return True
        else:
            print(f"Calibration duplication appeared to succeed, but {expected_calib_file_path} not found.")
            return False
    except Exception as e:
        print(f"Error duplicating calibration files to {calib_dest_path}: {e}")
        return False