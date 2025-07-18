import os
import shutil

from tqdm import tqdm
from datetime import datetime
import subprocess

FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"

def vd_separ_calib(calib_dir, calibVid, numViews):
    
    try:
        ffprobe_cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            calibVid
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

        ffmpeg_cmds = []
        skipped_views = []
        outputDirs = []

        for i in range(numViews):
            output_dir = os.path.join(calib_dir, f"view{i+1}", "calibration_images")
            outputDirs.append(output_dir)
            os.makedirs(output_dir, exist_ok=True)

        for i in range(numViews):
            row = i // 2
            col = i % 2
            x_offset = col * view_width
            y_offset = row * view_height
            output_file = os.path.join(outputDirs[i], f"SA_calib-view{i+1}-calibration.mp4")
            
            if os.path.exists(output_file):
                print(f"{output_file} already exists. Skipping view{i+1}")
                skipped_views.append(i)
            else:
                cmd = [
                    FFMPEG_PATH,
                    "-i", calibVid,
                    "-filter:v", f"crop={view_width}:{view_height}:{x_offset}:{y_offset}",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-progress", "pipe:1",
                    "-nostats",
                    output_file,
                ]
                ffmpeg_cmds.append(cmd)

        print("Starting video separation for calibration...")
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
                            # Extract time string (e.g., "00:00:15.12")
                            time_str = line.split("time=")[1].split(" ")[0]
                            h, m, s = map(float, time_str.split(':'))
                            current_time_seconds = h * 3600 + m * 60 + s
                            pbar.update(current_time_seconds - pbar.n) # Update with the difference
                        except (IndexError, ValueError):
                            pass # Ignore lines that don't parse as expected
            process.wait() # Wait for the process to complete

        print("Video separation complete.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except FileNotFoundError:
        print("FFmpeg or ffprobe not found. Make sure they are installed and in your PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    projectDir = 'D:/DGH/Data/Videos/2025-07-14 7day Marathon/'
    calibVideo = '2025-07-16-first3h.mkv'

    input_video = os.path.join(projectDir, calibVideo)

    num_camera_views = 4

    vd_separ_calib(projectDir, input_video, num_camera_views)
