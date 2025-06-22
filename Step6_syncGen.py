import os, sys
import subprocess
import scipy.io as sio
import numpy as np

# --- Project parameters, change it to the location of your sdannce project for Label3D.
projectDir = 'D:/Data/DGH/3D/20250517'

FFPROBE_PATH = "ffprobe" 

def get_cam_count(base_folder):
    if not os.path.isdir(base_folder):
        print(f"Error: Camera videos folder not found at {base_folder}", file=sys.stderr)
        return 0
    count = 0
    for root, dirs, files in os.walk(base_folder):
        if root.split(os.sep)[-1].startswith('Camera'):
            if '0.mp4' in files:
                count += 1
    return count

def get_calib_count(base_folder):
    if not os.path.isdir(base_folder):
        print(f"Error: Calibration folder not found at {base_folder}", file=sys.stderr)
        return 0
    count = 0
    for files in os.listdir(base_folder):
        if files.startswith('hires_cam'):
            count += 1
    return count

def get_frame_count(video_path):
    command = [
        FFPROBE_PATH,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f'Frame counts = {result.stdout.strip()}')
        return int(result.stdout.strip())
    except FileNotFoundError:
        print(f"Error: 'ffprobe' command not found. Make sure FFmpeg is installed and in your PATH.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe for {video_path}: {e.stderr}", file=sys.stderr)
        return None
    except ValueError:
        print(f"Error: Could not parse frame count for {video_path}.", file=sys.stderr)
        return None

numCam = get_cam_count(os.path.join(projectDir, 'Videos'))
print(f'Found {numCam} camera views.')
numCalib = get_calib_count(os.path.join(projectDir, 'Calibration'))
print(f'Found {numCalib} calibration files.')

if numCam == 0:
    print(f"Error: no camera views found in {projectDir}.")
    sys.exit(1)
elif numCalib == 0:
    print(f"Error: no calibration files found in {projectDir}.")
    sys.exit(1)
elif numCam != numCalib:
    print(f'Error: calibration and camera views do not match!')
    sys.exit(1)
else:

    videosample = os.path.join(projectDir, 'Videos', 'Camera1', '0.mp4')
    print('---Getting the frame counts from the first camera view---')
    framecount = get_frame_count(videosample)

    camnames = [f"Camera{i+1}" for i in range(numCam)]
    params = []
    sync = []

    for i in range(numCam):
            
            print(f'---Processing {camnames[i]}---')
            data_2d = np.zeros((framecount, 4))
            data_3d = np.zeros((framecount, 6))
            data_frame = np.arange(framecount, dtype=np.float64)
            data_sampleID = np.arange(framecount, dtype=np.float64)

            cam_struct = {
                    'data_2d': data_2d,
                    'data_3d': data_3d,
                    'data_frame': data_frame,
                    'data_sampleID': data_sampleID
                }
            
            calibFile = os.path.join(projectDir, 'Calibration', f'hires_cam{i+1}_params.mat')
            params_struct = sio.loadmat(calibFile)

            sync.append(cam_struct)
            params.append(params_struct)

    camnames_array = np.array(camnames, dtype=object)
    params_array = np.array(params, dtype=object).reshape(numCam, 1)
    sync_array = np.array(sync, dtype=object).reshape(numCam, 1)

    mat_data_to_save = {
        'camnames': camnames_array,
        'params': params_array,
        'sync': sync_array
        }
    output_mat_filename = 'sync-dannce.mat'
    output_mat_path = os.path.join(projectDir, output_mat_filename)
    sio.savemat(output_mat_path, mat_data_to_save)
    print(f"\nAll camera sync data saved to: {output_mat_path}")
