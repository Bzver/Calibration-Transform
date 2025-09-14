import os
import tkinter as tk
from tkinter import filedialog
import utils.clb_separator as cbs

def folder_select_dialog():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def collect_all_videos_in_folder(folder_path):
    video_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    return video_files

if __name__ == "__main__":
    folder_path = folder_select_dialog()
    video_files = collect_all_videos_in_folder(folder_path)
    for video_file in video_files:
        video_filename = os.path.basename(video_file).split(".")[0]
        output_dir = os.path.join(folder_path, f"{video_filename}")
        os.makedirs(output_dir, exist_ok=True)
        cbs.separate_video_stream(num_view=4, video_filepath=video_file, project_dir=output_dir, mode="calibration", use_gpu=True)

