import subprocess
import os
import datetime
import shutil

projectDir = 'D:/Project/Sleap-Models/3dT'
calibVideo = '2025-05-17 14-22-55.mkv'

input_video = os.path.join(projectDir, calibVideo)

# Number of camera views (can be 2, 3, 4, 5, or 6)
num_camera_views = 4 # Default to 4 as per current script behavior

FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"

current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S")

def vd_separ(input_video_path, num_views, output_dirs):

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

        # Assuming views are arranged in a x by 2 fashion (2 views a row)
        view_width = width // 2
        num_rows = num_views // 2 + 1 if num_views % 2 > 0 else num_views // 2
        view_height = height // num_rows

        ffmpeg_cmds = []
        for i in range(num_views):
            row = i // 2
            col = i % 2
            x_offset = col * view_width
            y_offset = row * view_height
            output_file = os.path.join(output_dirs[i], f"{formatted_datetime}-view{i+1}-calibration.mp4")

            cmd = [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:{x_offset}:{y_offset}",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                output_file,
            ]
            ffmpeg_cmds.append(cmd)

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

# Create the subfolders for each view inside the main output directory and collect their paths
output_directories = []
for i in range(1, num_camera_views + 1):
    output_dir = os.path.join(main_output_dir, f"view{i}", "calibration_images")
    output_directories.append(output_dir)
    os.makedirs(output_dir, exist_ok=True)

vd_separ(input_video, num_camera_views, output_directories)
