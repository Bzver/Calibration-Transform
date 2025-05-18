import os
import subprocess

projectDir = 'D:/Project/Sleap-Models/3dT'
behaVideo = '2025-05-17 20-47-47_segment_1.mp4'

input_video = os.path.join(projectDir, behaVideo)

FFMPEG_PATH = "ffmpeg"  
FFPROBE_PATH = "ffprobe" 

def vd_separ_2by2_2(input_video_path):

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
        view_width = width // 2
        view_height = height // 2

        # Determine base output directory
        input_video_dir = os.path.dirname(input_video_path)
        main_output_folder = input_video_dir

        # Initialize folder structure
        cam1Dir = os.path.join(main_output_folder, 'Videos', 'Camera1')
        cam2Dir = os.path.join(main_output_folder, 'Videos', 'Camera2')
        cam3Dir = os.path.join(main_output_folder, 'Videos', 'Camera3')
        cam4Dir = os.path.join(main_output_folder, 'Videos', 'Camera4')
        camDirAll = [cam1Dir, cam2Dir, cam3Dir, cam4Dir]
        for camDir in camDirAll:
            os.makedirs(camDir, exist_ok=True)

        # ffmpeg commands for each view (no audio)
        ffmpeg_cmds = [
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:0:0",
                "-c:v", "libx264", # Top-left
                "-preset", "slow",
                "-crf", "18",
                os.path.join(cam1Dir, "0.mp4"),
            ],
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:{view_width}:0", # Top-right
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                os.path.join(cam2Dir, "0.mp4"),
            ],
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:0:{view_height}", # Bottom-left
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                os.path.join(cam3Dir, "0.mp4"),
            ],
            [
                FFMPEG_PATH,
                "-i", input_video_path,
                "-filter:v", f"crop={view_width}:{view_height}:{view_width}:{view_height}", # Bottom-right
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                os.path.join(cam4Dir, "0.mp4"),
            ],
        ]

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

vd_separ_2by2_2(input_video)