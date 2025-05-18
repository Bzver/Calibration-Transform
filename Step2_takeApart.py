import subprocess
import os
import datetime
import glob
import shutil

projectDir = 'D:/Project/Sleap-Models/3dT'
calibVideo = '2025-05-17 14-22-55.mkv'

input_video = os.path.join(projectDir, calibVideo)

FFMPEG_PATH = "ffmpeg"  
FFPROBE_PATH = "ffprobe" 

current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

def vd_separ_2by2(input_video_path, output_dir1, output_dir2, output_dir3, output_dir4):

    os.chdir(os.path.dirname(os.path.abspath(__file__))) 

    try:
        ffprobe_cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_video_path
        ]
        ffprobe_output = subprocess.check_output(ffprobe_cmd).decode("utf-8").strip().split('\n')
        width, height, fps_str = ffprobe_output

        width = int(width)
        height = int(height)
        fps = eval(fps_str)
        view_width = width // 2
        view_height = height // 2

        ffmpeg_cmds = [
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:0:0", # Top-left
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                os.path.join(output_dir1, f"{formatted_datetime}-view1-calibration.mp4"),
            ],
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:{view_width}:0", # Top-right
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                os.path.join(output_dir2, f"{formatted_datetime}-view2-calibration.mp4"),
            ],
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:0:{view_height}", # Bottom-left
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                os.path.join(output_dir3, f"{formatted_datetime}-view3-calibration.mp4"),
            ],
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:{view_width}:{view_height}", # Bottom-right
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                os.path.join(output_dir4, f"{formatted_datetime}-view4-calibration.mp4"),
            ],
        ]

        for cmd in ffmpeg_cmds:
            subprocess.run(cmd, check=True)

        print("Video separation complete.")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except FileNotFoundError:
        print("FFmpeg or ffprobe not found. Make sure they are installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Create the main output directory with the formatted datetime
main_output_dir = os.path.join(projectDir, formatted_datetime)
os.makedirs(main_output_dir, exist_ok=True)

# Copy board.jpg and board.toml to the main output directory
board_jpg_path = os.path.join(projectDir, "board.jpg")
board_toml_path = os.path.join(projectDir, "board.toml")

if os.path.exists(board_jpg_path):
    shutil.copy(board_jpg_path, main_output_dir)
    print(f"Copied board.jpg to {main_output_dir}")
else:
    print(f"board.jpg not found in {projectDir}")

if os.path.exists(board_toml_path):
    shutil.copy(board_toml_path, main_output_dir)
    print(f"Copied board.toml to {main_output_dir}")
else:
    print(f"board.toml not found in {projectDir}")

# Create the subfolders for each view inside the main output directory
output_directory1 = os.path.join(main_output_dir, "view1", "calibration_images")
output_directory2 = os.path.join(main_output_dir, "view2", "calibration_images")
output_directory3 = os.path.join(main_output_dir, "view3", "calibration_images")
output_directory4 = os.path.join(main_output_dir, "view4", "calibration_images")

# Create the output directories if they don't exist
os.makedirs(output_directory1, exist_ok=True)
os.makedirs(output_directory2, exist_ok=True)
os.makedirs(output_directory3, exist_ok=True)
os.makedirs(output_directory4, exist_ok=True)

vd_separ_2by2(input_video, output_directory1, output_directory2, output_directory3, output_directory4)
