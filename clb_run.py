import os
import shutil
import sys

from datetime import datetime

from typing import Union

import utils.clb_helper as cbh
import utils.clb_separator as cbs
import utils.clb_transform as cbt
import utils.slap_functions as cbsp

ROOTPATH = "D:/Project/SDANNCE-Models/666-6CAM/"
CAMVIEWS = 6  # Number of camera views (can be 2, 3, 4, 5, or 6)
EXP = ""
CALIB = "D:/Project/SDANNCE-Models/666-6CAM/2025-09-10 15-31-15.mkv"
MERGED_VIDEO = True
EXCLUDE_CAM_ROTATE = [4,5]

def determine_project_dir(root_path:str, new_project:bool=False) -> Union[str, None]:
    """Determines the project directory to use based on existing folders or user input."""
    if not os.path.isdir(root_path):
        print(f"Error: {root_path} does not exist!")
        return

    project_name = None

    sd_folder_list = [ rpf for rpf in os.listdir(root_path) if rpf.startswith ("SD-") ]
    
    if any([not sd_folder_list, new_project]):
        project_dir = determine_new_project_dir(root_path)
        return project_dir

    if len(sd_folder_list) == 1:
        project_name = str(sd_folder_list[0])
        print(f"Project found! Loading {project_name}...")
        return os.path.join(root_path, project_name)

    print(f"Multiple project folders found. Choose which to load.")
    for i, folder in enumerate(sd_folder_list, 1):
        print(f"{i}: {folder}")

    while project_name is None:
        try:
            choice = int(input("Enter selection number: ")) - 1
            if 0 <= choice < len(sd_folder_list):
                project_name = str(sd_folder_list[choice])
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input (must be a number)")
        except KeyboardInterrupt:
            print("\nProject selection cancelled. Exiting.")
            sys.exit("User cancelled project selection.")
    return os.path.join(root_path, project_name)

def determine_new_project_dir(root_path:str) -> str:
    """Creates a new project directory with an optional custom name. """
    current_time = datetime.now()
    time_formatted = current_time.strftime("%Y%m%d%H%M%S")

    custom_name = input("Custom name for new project directory (leave blank for timestampe name): ").strip()
    temp_name = f"SD-{custom_name}" if custom_name else f"SD-{time_formatted}"
    base_name = temp_name
    counter = 1
    
    # Handle name collisions for the new project
    while os.path.isdir(os.path.join(root_path, temp_name)):
        print(f"Warning: Project folder '{temp_name}' already exists.")
        temp_name = f"{base_name}({counter})"
        counter += 1
  
    project_dir = os.path.join(root_path, temp_name)
    os.makedirs(project_dir, exist_ok=True)
    print(f"Determined new project name: {project_dir}")
    return project_dir

def load_video(num_view:int, video_path:str, dir_path:str, mode, merged_video_stream:bool=False):
    """
    Handles loading and preparing video files for calibration or experiment.

    Args:
        num_view (int): The number of camera views expected.
        video_path (str): The path to the video file or directory.
        dir_path (str): The destination directory path for the processed videos.
        mode (Literal["calibration", "experiment"]): The processing mode.
        merged_video_stream (bool): True if a single video file contains all streams.
                                    False if videos for each view are in a directory.

    Returns:
        bool: True if the videos are successfully processed, False otherwise.
    """
    if merged_video_stream:
        if not os.path.isfile(video_path):
            print(f"Error: Input video path is not a file: {video_path} while merged_video_stream is True.")
            return False
    
        if not cbs.separate_video_stream(num_view, video_path, dir_path, mode):
            return False

    else:
        if not os.path.isdir(video_path):
            print(f"Error: Input video path is not a directory: {video_path} while merged_video_stream is False.")
            return False
        
        video_list, video_count = cbh.get_eligible_video_in_folder(video_path)

        if not video_list or video_count != num_view:
            print(f"The number of eligible videos ({video_count}) does not match the expected number of views ({num_view}).")
            return False
        
        if not cbh.create_output_dirs(num_view, dir_path, mode):
            return False
        
        for i, video in enumerate(video_list):
            if mode == "calibration":
                video_dest = os.path.join(dir_path, f"view{i+1}", "calibration_images")
                shutil.copy2(video, os.path.join(video_dest, f"SA_calib-view{i+1}-calibration.mp4"))
            elif mode == "experiment":
                video_dest = os.path.join(dir_path, "Videos", f"Camera{i+1}")
                shutil.copy2(video, os.path.join(video_dest, "0.mp4"))
            else:
                print("Invalid mode. Expected 'calibration' or 'experiment'.")
                return False

    return True

def create_new_project(
        num_view:int,
        root_path:str,
        exp_video_path:str,
        calib_video_path:str,
        merge_video_stream:bool=False,
        calibration_only:bool=False
        ) -> bool:

    if not calibration_only:
        project_dir = determine_project_dir(root_path, new_project=True)
        if not project_dir:
            return False
        load_video(num_view, exp_video_path, project_dir, "experiment", merged_video_stream=merge_video_stream)

    calib_dir = os.path.join(root_path, "SA_calib")
    os.makedirs(calib_dir, exist_ok=True)
    calib_lost_file = cbh.check_calib_integrity(root_path, num_view)

    if calib_lost_file:
        if any(["calibration.toml" in f for f in calib_lost_file]):
            load_video(num_view, calib_video_path, calib_dir, "calibration", merged_video_stream=merge_video_stream)
            cbsp.generate_calib_board(calib_dir)
            if not cbsp.calibration(calib_dir):
                print("Calibration failed!")
                return False

        if not cbt.process_sleap_calibration(calib_dir, excluded_cams_for_rotation=EXCLUDE_CAM_ROTATE):
            print("Calibration transformation failed!")
            return False

    if not calibration_only:
        if not cbh.check_project_integrity(num_view, root_path, project_dir):
            return False
    
    return True

def load_existing_project(num_view:int, root_path:str) -> bool:
    project_dir = determine_project_dir(root_path, new_project=False)
    if not project_dir:
        return False

    if not cbh.check_project_integrity(num_view, root_path, project_dir):
        return False
    
    print("Existing project loaded and checked successfully.")
    return True

if __name__ == "__main__":
    if not EXP:
        calibration_only = True
    else:
        calibration_only = False

    calib_vid_path = os.path.join(ROOTPATH, CALIB)
    if create_new_project(
        num_view=CAMVIEWS,
        root_path=ROOTPATH,
        exp_video_path=EXP,
        calib_video_path=CALIB,
        merge_video_stream=MERGED_VIDEO,
        calibration_only=calibration_only
        ):
        print("Success.")
    else:
        print("Failure.")
