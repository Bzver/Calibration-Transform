import os
import subprocess

projectDir = 'D:/Project/Sleap-Models/3dT'
behaVideo = '2025-05-17 20-47-47_segment_1.mp4'

input_video = os.path.join(projectDir, behaVideo)

# Number of camera views (can be 2, 3, 4, 5, or 6)
num_camera_views = 4 # Default to 4 as per current script behavior

FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"
def vd_separ_2(input_video_path):
    global num_camera_views # Declare use of global variable

    try:
        if not os.path.isfile(input_video_path):
            print(f"Error: Input video path is not a file: {input_video_path}")
            return
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

        # Assuming views are arranged in a x by 2 fashion (2 views a row)
        view_width = width // 2
        num_rows = num_camera_views // 2 + 1 if num_camera_views % 2 > 0 else num_camera_views // 2
        view_height = height // num_rows

        # Determine base output directory
        input_video_dir = os.path.dirname(input_video_path)
        main_output_folder = input_video_dir

        # Initialize folder structure
        cam_dirs = []
        for i in range(1, num_camera_views + 1):
            cam_dir = os.path.join(main_output_folder, 'Videos', f'Camera{i}')
            cam_dirs.append(cam_dir)
            os.makedirs(cam_dir, exist_ok=True)

        # ffmpeg commands for each view (no audio)
        ffmpeg_cmds = []
        for i in range(num_camera_views):
            row = i // 2
            col = i % 2
            x_offset = col * view_width
            y_offset = row * view_height
            output_file = os.path.join(cam_dirs[i], "0.mp4")

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

        # Execute ffmpeg commands
        for cmd in ffmpeg_cmds:
            subprocess.run(cmd, check=True)

        print("Video separation complete.")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except FileNotFoundError:
        print("FFmpeg or ffprobe not found. Make sure they are installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")

vd_separ_2(input_video)