This repo contains a pipeline for making use of sleap-anipose's 3D calibration toolkit to generate calibration files complatible with Label3D (Required for DANNCE and SDANNCE). I made this mainly because I just couldn't wrap my head around how DANNCE's own calibration tool works and intended on using the intuitive and well-documented sleap-anipose triangulation and calibration toolbox instead.

**Sleap-anipose:** 
https://github.com/talmolab/sleap-anipose

**DANNCE and SDANNCE:**

https://github.com/spoonsso/dannce/

https://github.com/tqxli/sdannce

**Label3D:**
https://github.com/diegoaldarondo/Label3D

## 1. [`Step1_boardGen.py`]

**Purpose:** Generates an ArUco calibration board image and its corresponding configuration file.

**Usage:**
This script uses the `sleap_anipose` library to create a calibration board. Set the `rootpath` to where you want to create the project for `sleap_anipose` calibration

**Parameters:**
Check the sleap-anipose documentation for how to adjust these parameters.
https://github.com/talmolab/sleap-anipose/blob/main/docs/CALIBRATION_GUIDE.md

**Output:**
-   A JPEG image of the calibration board (`board.jpg`).
-   A TOML configuration file describing the board (`board.toml`).

## 2. [`Step2_takeApart.py`]

**Purpose:** Splits a single calibration video (in my case obtained from OBS for camera syncing) containing multiple camera views (expected in a 2x2 grid) into separate video files for each view. It also copies the generated `board.jpg` and `board.toml` files to a new timestamped directory.

**Usage:**
This script requires FFmpeg and FFprobe to be installed and accessible in your system's PATH.

1.  **Configure `projectDir`**: Set the `projectDir` to the same as the previous `rootpath`.
2.  **Configure `calibVideo`**: Set the `calibVideo` variable to the filename of your calibration video within the `projectDir`.
3.  **Run the script**: Execute the Python script.

**Output:**
-   A new directory within `projectDir` named with the current timestamp (e.g., `D:/Project/Sleap-Models/3dT/20250518214656`).
-   Copies `board.jpg` and `board.toml` into the new timestamped directory.
-   Creates subdirectories `view1/calibration_images`, `view2/calibration_images`, `view3/calibration_images`, and `view4/calibration_images` within the timestamped directory.
-   Saves the separated calibration videos (e.g., `20250518214656-view1-calibration.mp4`) into their respective `calibration_images` subdirectories.

## 3. [`Step3_3dCalib.py`]

**Purpose:** Performs the 3D camera calibration using the separated calibration videos and the board configuration file.

**Usage:**
This script uses the `sleap_anipose` library for calibration.

1.  **Configure `rootpath`**
2.  **Ensure calibration images are ready**: Make sure the separated calibration videos from Step 2 are in the `calibration_images` subdirectories within the latest timestamped project directory.
3.  **Run the script**: Execute the Python script.

**Output:**
-   A `calibration.toml` file containing the intrinsic and extrinsic parameters for each camera, saved in the latest timestamped project directory.
-   A `calibration.metadata.h5` file containing calibration metadata.
-   A `reprojection_histogram.png` image visualizing reprojection errors.
-   Other potential output files related to the calibration process within the session directory.

## 4. [`Step4_calibTF.py`]

**Purpose:** Transforms the camera calibration parameters obtained from sleap-aniposse/Anipose (`calibration.toml`) into a format suitable for Label3D (`.mat` files). Also visualizes the original and transformed camera positions and orientations for verifications.

**Usage:**
This script requires `numpy`, `matplotlib`, `scipy`, and `Python 3.11` or higher.

1.  **Configure `projectDir`**: Set the `projectDir` variable (line 10) to the specific timestamped directory containing the `calibration.toml` file generated in Step 3.
2.  **Run the script**: Execute the Python script.

**Output:**
-   Prints relative geometry statistics for the original and transformed camera setups.
-   Generates `.mat` files (e.g., `hires_cam1_params.mat`) for each camera, containing the transformed intrinsic (`K`), rotation (`r`), translation (`t`), and distortion (`RDistort`, `TDistort`) parameters. These are saved in the `projectDir`.
-   Displays 3D plots visualizing the original and transformed camera positions and orientations, along with the calculated intersection points and a ground plane.

## 5. [`Step5_takeApart2ElectricBoogaloo.py`]

**Purpose:** Splits the behavior video in the same fashion as Step2, albeit organized into a 'Videos' subdirectory structure for Label3D.

**Usage:**
This script requires FFmpeg and FFprobe to be installed and accessible in your system's PATH.

**Parameters:**
-   `projectDir`: The base directory containing the behavior video.
-   `behaVideo`: The filename of the behavior video (expected to be in a 2x2 grid layout).

**Output:**
-   Creates a `Videos` subdirectory within `projectDir`.
-   Creates `Camera1`, `Camera2`, `Camera3`, and `Camera4` subdirectories within the `Videos` directory.
-   Saves the separated behavior videos (named `0.mp4`) into their respective `CameraX` subdirectories.

## 6. [`Step6_syncGen.py`]

**Purpose:** Generates a `sync-dannce.mat` file required by Label3D, combining information about camera names, calibration parameters (from the `.mat` files generated in Step 4), and camera sync data.

**Usage:**
This script requires `scipy.io` and `numpy`.

1.  **Configure `projectDir`**: Set the `projectDir` variable (line 7) to the root directory of your DANNCE project. This directory should contain the `Videos` subdirectory (copy from Step 5) and the `Calibration` subdirectory (copy from Step 4). You need to do these copy work manually!
2.  **Ensure files are in place**: Make sure the `Videos` directory with separated `0.mp4` files and the `Calibration` directory with `hires_camX_params.mat` files are correctly placed within the `projectDir`.
3.  **Run the script**: Execute the Python script.

**Output:**
-   Validates the number of camera video directories and calibration files.
-   Generates a `sync-dannce.mat` file in the `projectDir`, containing the camera names, calibration parameters, and initialized data arrays. This file is used by Label3D for 3D reconstruction.
