import os

import tqdm
import subprocess

FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"

def vd_separ_exp(behVid, project_dir, numViews):

    try:
        if not os.path.isfile(behVid):
            print(f"Error: Input video path is not a file: {behVid}")
            return
        ffprobe_cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            behVid
        ]
        ffprobe_output = subprocess.check_output(ffprobe_cmd).decode("utf-8").strip().split('\n')
        width, height, duration_str = ffprobe_output
        width = int(width)
        height = int(height)
        total_duration = float(duration_str)

        # Assuming views are arranged in a x by 2 fashion (2 views a row)
        view_width = width // 2
        num_rows = numViews // 2 + 1 if numViews % 2 > 0 else numViews // 2
        view_height = height // num_rows
        main_output_folder = project_dir

        # Initialize folder structure
        cam_dirs = []
        for i in range(1, numViews + 1):
            cam_dir = os.path.join(main_output_folder, 'Videos', f'Camera{i}')
            cam_dirs.append(cam_dir)
            os.makedirs(cam_dir, exist_ok=True)

        # ffmpeg commands for each view
        ffmpeg_cmds = []
        for i in range(numViews):
            row = i // 2
            col = i % 2
            x_offset = col * view_width
            y_offset = row * view_height
            output_file = os.path.join(cam_dirs[i], "0.mp4")
            skipped_views = []
            if os.path.exists(output_file):
                print(f"{output_file} already exists. Skipping view{i+1}")
                skipped_views.append(i)
            else:
                cmd = [
                    FFMPEG_PATH,
                    "-i", behVid,
                    "-filter:v", f"crop={view_width}:{view_height}:{x_offset}:{y_offset}",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-progress", "pipe:1",
                    "-nostats",
                    output_file,
                ]
                ffmpeg_cmds.append(cmd)

        print("Starting video separation for experiment...")
        for i, cmd in enumerate(ffmpeg_cmds):
            current_view = i+1
            while current_view in skipped_views:
                current_view +=1 
            print(f"\nProcessing View {i+1}/{numViews-len(skipped_views)}")
            # Start subprocess with stdout as PIPE to capture progress
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

            with tqdm(total=int(total_duration), unit="s", desc=f"View {current_view}") as pbar:
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

if __name__ == "__main__":
    projectDir = 'D:/Project/Sleap-Models/3dT'
    behaVideo = '2025-05-17 20-47-47_segment_1.mp4'

    input_video = os.path.join(projectDir, behaVideo)

    # Number of camera views (can be 2, 3, 4, 5, or 6)
    num_camera_views = 4 # Default to 4 as per current script behavior

    vd_separ_exp(input_video)