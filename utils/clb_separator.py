import os
import subprocess

from tqdm import tqdm

from typing import List, Tuple
from .clb_helper import create_output_dirs

def separate_video_stream(num_view:int, video_filepath:str, project_dir:str, mode:str, use_gpu:bool) -> bool:
    try:
        if not os.path.isfile(video_filepath):
            print(f"Error: Input video path is not a file: {video_filepath}")
            return False

        width, height, total_duration = acquire_video_metadata(video_filepath)

        # Assuming views are arranged in a x by 2 fashion (2 views a row)
        view_width = width // 2
        num_rows = num_view // 2 + 1 if num_view % 2 > 0 else num_view // 2
        view_height = height // num_rows
        view_parameters = (view_width, view_height)

        output_dir_list = create_output_dirs(num_view, project_dir, mode)
        ffmpeg_cmds, skipped_views = assemble_ffmpeg_commands(num_view, view_parameters, video_filepath, output_dir_list, mode, use_gpu)

        print("Starting video separation for calibration...")
        for i, cmd in enumerate(ffmpeg_cmds):
            current_view = i+1
            while current_view in skipped_views:
                current_view +=1 
            print(f"\nProcessing View {i+1}/{num_view-len(skipped_views)}")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

            with tqdm(total=int(total_duration), unit="s", desc=f"View {current_view}") as pbar: # Progress bar functionality
                for line in process.stdout:
                    if "time=" in line:
                        try:
                            time_str = line.split("time=")[1].split(" ")[0] 
                            h, m, s = map(float, time_str.split(':'))
                            current_time_seconds = h * 3600 + m * 60 + s
                            pbar.update(current_time_seconds - pbar.n)
                        except (IndexError, ValueError):
                            pass
            process.wait()

        print("Video separation complete.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except FileNotFoundError:
        print("FFmpeg or ffprobe not found. Make sure they are installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

def acquire_video_metadata(video_filepath:str) -> Tuple[int, int, float]:
    try:
        ffprobe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_filepath
        ]
        ffprobe_output = subprocess.check_output(ffprobe_cmd).decode("utf-8").strip().split('\n')
        width, height, duration_str = ffprobe_output
        width = int(width)
        height = int(height)
        total_duration = float(duration_str)
        return width, height, total_duration
    except subprocess.CalledProcessError as e:
        print(f"FFprobe error: {e}")
    except FileNotFoundError:
        print("FFprobe not found. Make sure it is installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

def assemble_ffmpeg_commands(
        num_view: int,
        view_parameters: Tuple[int, int],
        video_filepath: str,
        output_dir_list: List[str],
        mode: str,
        use_gpu: bool = False
        ) -> Tuple[List[List[str]], List[int]]:
    ffmpeg_cmds, skipped_views = [], []

    for i in range(num_view):
        row = i // 2
        col = i % 2
        x_offset = col * view_parameters[0]
        y_offset = row * view_parameters[1]

        if mode == "calibration":
            output_filepath = os.path.join(output_dir_list[i], f"SA_calib-view{i+1}-calibration.mp4")
        elif mode == "experiment":
            output_filepath = os.path.join(output_dir_list[i], "0.mp4")
        else:
            print("Invalid mode. Expected 'calibration' or 'experiment'.")
            return ffmpeg_cmds, skipped_views

        if os.path.isfile(output_filepath):
            print(f"{output_filepath} already exists. Skipping view{i+1}")
            skipped_views.append(i)
        else:
            cmd = ["ffmpeg"]
            if use_gpu:
                cmd.extend([
                    "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda",
                    "-i", video_filepath,
                    "-vf", f"hwdownload,format=nv12,crop={view_parameters[0]}:{view_parameters[1]}:{x_offset}:{y_offset},hwupload_cuda",
                    "-c:v", "h264_nvenc",
                    "-preset", "p7",
                    "-global_quality", "18",
                    "-rc", "vbr_hq",
                ])
            else:
                cmd.extend([
                    "-i", video_filepath,
                    "-filter:v", f"crop={view_parameters[0]}:{view_parameters[1]}:{x_offset}:{y_offset}",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                ])

            cmd.extend(["-progress", "pipe:1", "-nostats"])
            cmd.append(output_filepath)
            ffmpeg_cmds.append(cmd)

    return ffmpeg_cmds, skipped_views