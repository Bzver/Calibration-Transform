# Calibration-Transform

This repository provides a Python script for performing camera calibration and preparing video data for multi-view animal pose estimation. It integrates with `sleap-anipose` for calibration and prepares data for `DANNCE` and `SDANNCE`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/Calibration-Transform.git
    cd Calibration-Transform
    ```

2.  **Install Dependencies:**
    -   Follow the [sleap-anipose installation guide](https://github.com/talmolab/sleap-anipose/blob/main/README.md) to set up `sleap-anipose` and its dependencies.
    -   Ensure `ffmpeg` and `ffprobe` are installed and accessible in your system's PATH. These are required for video stream separation.

## Usage

The primary entry point for this tool is the `calib_run.py` script. This script handles project setup, video processing, and calibration.

### Configuration in `calib_run.py`

Before running, you need to configure the following variables in `calib_run.py`:

-   `ROOTPATH`: The base directory where your project folders (`SD-` prefixed) and calibration data (`SA_calib`) will be stored or are located.
    Example: `ROOTPATH = "D:/Project/SDANNCE-Models/4CAM-3D-2ETUP"`

-   `CAMVIEWS`: The number of camera views (e.g., 2, 3, 4, 5, or 6) used in your setup. This must match the number of physical cameras.
    Example: `CAMVIEWS = 4`

-   `EXP`: The path to your experiment video(s).
    -   If `MERGED_VIDEO` is `True`, this should be the path to a single merged video file (e.g., from OBS).
    -   If `MERGED_VIDEO` is `False`, this should be the path to a directory containing individual video files for each camera view (e.g., `Camera1/0.mp4`, `Camera2/0.mp4`, etc.).
    Example: `EXP = os.path.join(ROOTPATH, "20250620-76225401-03-processed.mp4")`

-   `CALIB`: The path to your calibration video(s).
    -   Similar to `EXP`, this can be a single merged video or a directory of individual calibration videos, depending on `MERGED_VIDEO`.
    Example: `CALIB = os.path.join(ROOTPATH, "2025-05-17 14-22-55.mkv")`

-   `MERGED_VIDEO`: A boolean flag. Set to `True` if your `EXP` and `CALIB` videos are single files containing merged streams from multiple cameras. Set to `False` if they are directories containing separate video files for each camera.
    Example: `MERGED_VIDEO = True`

### Running the Script

To run the script, navigate to the `Calibration-Transform` directory in your terminal and execute:

```bash
python calib_run.py
```

The script will guide you through creating a new project or loading an existing one, processing videos, and performing calibration.

## Folder Structure

The tool expects and generates a specific folder structure within your `ROOTPATH`:

```
ROOTPATH/
├── SA_calib/
│   ├── board.jpg
│   ├── board.toml
│   ├── calibration.toml
│   ├── calibration.metadata.h5
│   ├── reprojection_histogram.png
│   └── Calibration/
│       ├── hires_cam1_params.mat
│       ├── hires_cam2_params.mat
│       └── ...
│       └── view1/
│           └── calibration_images/
│               └── SA_calib-view1-calibration.mp4
│       └── view2/
│           └── calibration_images/
│               └── SA_calib-view2-calibration.mp4
│       └── ...
└── SD-{PROJECT_NAME}/
    ├── Videos/
    │   ├── Camera1/
    │   │   └── 0.mp4
    │   ├── Camera2/
    │   │   └── 0.mp4
    │   └── ...
    ├── Calibration/
    │   ├── hires_cam1_params.mat
    │   ├── hires_cam2_params.mat
    │   └── ...
    └── sync_dannce.mat
```

-   `SA_calib/`: Contains all calibration-related files, including the generated calibration board, TOML configuration, and `.mat` files with transformed camera parameters.
-   `SD-{PROJECT_NAME}/`: Each project will have its own directory, prefixed with `SD-`. This folder contains the experiment videos, duplicated calibration files, and the `sync_dannce.mat` file required for DANNCE.

## Links

-   [Sleap-anipose](https://github.com/talmolab/sleap-anipose)
-   [DANNCE](https://github.com/spoonsso/dannce/)
-   [SDANNCE](https://github.com/tqxli/sdannce)
-   [Label3D](https://github.com/diegoaldarondo/Label3D)
