import os

import toml

import numpy as np
from itertools import combinations

import scipy.io as sio
import scipy.spatial.transform as st
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Any
from numpy.typing import NDArray

from clb_helper import check_dannce_mat

def process_sleap_calibration(calib_dir:str, show_plots:bool=True, save_mat_files:bool=True) -> bool:
    """
    Processes a SLEAP calibration file to transform camera parameters,
    visualize the setup, and export results to .mat files.

    Args:
        calib_dir (str): The directory containing the 'calibration.toml' file.
        show_plots (bool): If True, displays the 3D plots for visualization.
        save_mat_files (bool): If True, saves the transformed parameters to .mat files.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    calib_filepath = os.path.join(calib_dir, 'calibration.toml')
    
    original_params_list, camera_positions, camera_orientations, cam_count = extract_original_calibration(calib_filepath)
    if not original_params_list:
        return False
    
    P_intersect_orig = calculate_camera_intersect(camera_orientations, camera_positions)
    plane_rotation_matrix, centroid = calculate_plane_rotation(camera_positions)
    rel_dist_org, rel_ang_org = calculate_relative_geometry_stats(camera_positions, camera_orientations, "Pre-transpose")

    transformed_positions, transformed_orientations, final_P_intersect = apply_rotation(
        plane_rotation_matrix, centroid, P_intersect_orig, camera_positions, camera_orientations)
    rel_dist_post, rel_ang_post = calculate_relative_geometry_stats(transformed_positions, transformed_orientations, "Post-transpose")

    if rel_dist_org and rel_dist_post:
        if not compare_relative_geometries(rel_dist_org, rel_ang_org, rel_dist_post, rel_ang_post, new_label="Post-transpose"):
            print("Relative geometry mismatch from transpose detected, aborting...")
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            plot_relative_geometry(ax1, camera_positions, camera_orientations, cam_count, 'Original', P_intersect_orig)
            plot_relative_geometry(ax2, transformed_positions, transformed_orientations, cam_count, 'Rotated', final_P_intersect)
            return False
    
    if any([not check_dannce_mat(calib_dir), save_mat_files]):
        mat_save_folder = save_transformed_mat(calib_dir, original_params_list, transformed_positions, transformed_orientations)

    mat_cam_positions, mat_cam_orientations = load_and_validate_mat(cam_count, mat_save_folder)
    mat_intersect = calculate_camera_intersect(mat_cam_orientations, mat_cam_positions)
    rel_dist_mat, rel_ang_mat = calculate_relative_geometry_stats(mat_cam_positions, mat_cam_orientations, "Geometry from .mat Files")

    if rel_dist_org and rel_dist_mat:
        if not compare_relative_geometries(rel_dist_org, rel_ang_org, rel_dist_mat, rel_ang_mat, new_label=".mat"):
            print("Relative geometry mismatch from .mat detected, aborting...")
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            plot_relative_geometry(ax1, camera_positions, camera_orientations, cam_count, 'Original', P_intersect_orig)
            plot_relative_geometry(ax2, mat_cam_positions, mat_cam_orientations, cam_count, ".mat", mat_intersect)
            return False

    if show_plots:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        plot_relative_geometry(ax1, camera_positions, camera_orientations, cam_count, 'Original', P_intersect_orig)
        plot_relative_geometry(ax2, transformed_positions, transformed_orientations, cam_count, 'Rotated', final_P_intersect)
        plot_relative_geometry(ax3, mat_cam_positions, mat_cam_orientations, cam_count, ".mat", mat_intersect)

        plt.tight_layout()
        plt.show()

    return True

def extract_original_calibration(calib_filepath:str) -> Tuple[List[Dict[str, Any]], NDArray, NDArray, int]:
    if not os.path.isfile(calib_filepath):
        print(f"Error: Calibration file not found at {calib_filepath}")
        return False

    try:
        with open(calib_filepath, 'r') as f:
            clbf = toml.load(f)
    except Exception as e:
        print(f"Error reading or parsing TOML file: {e}")
        return False

    original_params_list = []
    camera_positions = []
    camera_orientations = []
    cam_count = 0

    for camera_data in clbf.values():
        if 'matrix' in camera_data:
            cam_count += 1
            
            # Extract original parameters from the TOML file
            K_orig = np.array(camera_data['matrix']).T
            rotations_orig_vec = np.array(camera_data['rotation'])
            R_orig = st.Rotation.from_rotvec(rotations_orig_vec).as_matrix()
            t_orig = np.array(camera_data['translation'])
            RDistort_orig = np.array(camera_data['distortions'][0:2])
            TDistort_orig = np.array(camera_data['distortions'][2:4])

            original_params_list.append({
                'K': K_orig, 'R_orig': R_orig,
                'RDistort': RDistort_orig, 'TDistort': TDistort_orig,
                't_orig': t_orig
            })

            # Calculate camera position and orientation in world coordinates
            cam_pos = -np.dot(R_orig.T, t_orig)
            cam_dir = R_orig[2, :]  # Principal axis from the 3rd row of rotation matrix

            camera_positions.append(cam_pos)
            camera_orientations.append(cam_dir)

    if not camera_positions:
        print("No camera data with a 'matrix' key found in the calibration file.")
        return False
        
    camera_positions = np.array(camera_positions)
    camera_orientations = np.array(camera_orientations)

    return original_params_list, camera_positions, camera_orientations, cam_count

def calculate_camera_intersect(cam_orientations, cam_positions):
    """
    Find the point of closest intersection of the camera orientation rays

    Args:
        cam_orientations (np.ndarray]): List of camera orientation vectors.
        cam_positions (np.ndarray]): List of camera position vectors.

    Returns:
        np.ndarray: The point of closest intersection.
    
    """
    A_lstsq = np.vstack([np.eye(3) - np.outer(d, d) for d in cam_orientations])
    b_lstsq = np.hstack([(np.eye(3) - np.outer(d, d)) @ p for p, d in zip(cam_positions, cam_orientations)])
    intersection = np.linalg.lstsq(A_lstsq, b_lstsq, rcond=None)[0]
    return intersection

def calculate_plane_rotation(camera_positions):
    """
    Calculates the rotation matrix needed to align a plane fitted to camera positions
    with the ground plane (XY-plane).

    This function first fits a plane to a set of 3D camera positions using Principal Component
    Analysis (PCA) to find the plane's normal vector. It then calculates the rotation required
    to make this normal vector align with the Z-axis ([0, 0, 1]), effectively rotating the
    camera plane to be parallel with the XY ground plane. It also returns the centroid of
    the camera positions, which can be used as a translation reference.

    Args:
        camera_positions (NDArray): A NumPy array of shape (N, 3), where N is the number of
                                     cameras. Each row represents the 3D position of a camera.

    Returns:
        Tuple[NDArray, NDArray]: A tuple containing:
            - plane_rotation_matrix (NDArray): A 3x3 rotation matrix that, when applied to
              the camera positions, aligns the fitted plane with the XY-plane.
            - centroid (NDArray): A 1D NumPy array representing the 3D centroid of the
              camera positions.
    """
    centroid = np.mean(camera_positions, axis=0)
    covariance_matrix = np.cov(camera_positions - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    if normal_vector[2] < 0:
        normal_vector *= -1  # Ensure normal points upwards

    # Calculate rotation to align the camera plane with the XY ground plane (normal = [0,0,1])
    ground_normal = np.array([0, 0, 1])
    rotation_vector = np.cross(normal_vector, ground_normal)
    rotation_vector_norm = np.linalg.norm(rotation_vector)
    
    rotation_axis = rotation_vector / rotation_vector_norm
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_angle = np.arccos(np.dot(normal_vector, ground_normal))
        rotation = st.Rotation.from_rotvec(rotation_axis * rotation_angle)
        plane_rotation_matrix = rotation.as_matrix()
    else:
        plane_rotation_matrix = np.eye(3)

    return plane_rotation_matrix, centroid

def apply_rotation(plane_rotation_matrix, centroid, intersection, camera_positions, camera_orientations):
    """Applies a series of transformations to camera positions and orientations."""
    # Translate intersection point to origin (Z=0) and rotate cameras around it
    rotated_intersect = np.dot(plane_rotation_matrix, (intersection - centroid))
    target_intersection_z = 0.0
    z_translation = target_intersection_z - rotated_intersect[2]

    transformed_positions = np.dot(plane_rotation_matrix, (camera_positions - centroid).T).T 
    transformed_positions[:, 2] += z_translation
    
    transformed_intersect = rotated_intersect + np.array([0, 0, z_translation])

    # Apply a 180-degree rotation around the X-axis to flip the coordinate system
    R_180_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    
    temp_transformed_positions = np.zeros_like(transformed_positions)
    for i in range(len(transformed_positions)):
        pos_relative_to_intersect = transformed_positions[i] - transformed_intersect
        rotated_pos_relative = R_180_x @ pos_relative_to_intersect
        temp_transformed_positions[i] = rotated_pos_relative + transformed_intersect
    transformed_positions = temp_transformed_positions

    transformed_orientations = []
    plot_cam_idx = 0 # Use a 0-based index for plotting transformed cameras

    original_dir_for_this_cam = camera_orientations[plot_cam_idx]
    intermediate_dir = np.dot(plane_rotation_matrix, original_dir_for_this_cam)
    transformed_dir = R_180_x @ intermediate_dir
    transformed_orientations.append(transformed_dir)
    plot_cam_idx +=1

    transformed_orientations = np.array(transformed_orientations)
    return transformed_positions, transformed_orientations, transformed_intersect

def calculate_pairwise_distances(positions):
    """Calculates pairwise Euclidean distances between camera positions."""
    num_cameras = len(positions)
    distances = {}
    if num_cameras < 2:
        return distances
    for i, j in combinations(range(num_cameras), 2):
        pair_key = f"Cam{i+1}-Cam{j+1}" # Using 1-based indexing for user-friendliness
        dist = np.linalg.norm(positions[i] - positions[j])
        distances[pair_key] = dist
    return distances

def calculate_pairwise_angles(orientations):
    """Calculates pairwise angles (in degrees) between camera orientation vectors."""
    num_cameras = len(orientations)
    angles = {}
    if num_cameras < 2:
        return angles
    for i, j in combinations(range(num_cameras), 2):
        pair_key = f"Cam{i+1}-Cam{j+1}"
        dir1 = orientations[i] / np.linalg.norm(orientations[i]) # Ensure normalized
        dir2 = orientations[j] / np.linalg.norm(orientations[j])
        dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0) # Clip for numerical stability
        angle_rad = np.arccos(dot_product)
        angles[pair_key] = np.degrees(angle_rad)
    return angles

def calculate_relative_geometry_stats(positions, orientations, label=""):
    """Prints relative distances and angles for a set of cameras."""
    print(f"\n--- Relative Geometry Statistics for {label} ---")
    if len(positions) < 2:
        print("  Not enough cameras to calculate relative geometry.")
        return None, None
        
    rel_distances = calculate_pairwise_distances(positions)
    rel_angles = calculate_pairwise_angles(orientations)
    
    for pair_key in rel_distances.keys():
        dist_str = f"{rel_distances[pair_key]:.2f}"
        angle_str = f"{rel_angles.get(pair_key, float('nan')):.2f} deg" if pair_key in rel_angles else "N/A"
        print(f"  {pair_key}: Distance = {dist_str}, Angle = {angle_str}")
    return rel_distances, rel_angles

def compare_relative_geometries(orig_distances, orig_angles, new_distances, new_angles, orig_label="Original", new_label="Transformed"):
    """Compares and prints relative geometries of two sets of cameras."""
    #print(f"\n--- Comparison: {orig_label} vs. {new_label} Relative Geometry ---")
    if orig_distances is None or not orig_distances:
        print(f"  Cannot perform comparison: Missing or empty {orig_label} data.")
        return False
    if new_distances is None or not new_distances:
        print(f"  Cannot perform comparison: Missing or empty {new_label} data.")
        return False

    all_pair_keys = sorted(list(set(orig_distances.keys()) | set(new_distances.keys())))
    diff_tolerance = 1e-9

    for pair_key in all_pair_keys:
        
        orig_dist = orig_distances.get(pair_key, float('nan'))
        new_dist = new_distances.get(pair_key, float('nan'))
        if abs(orig_dist - new_dist) > diff_tolerance:
            print(f"  Pair: {pair_key}")
            print(f"    Relative Distance Mismatch: {orig_label} = {orig_dist:.2f}, {new_label} = {new_dist:.2f}",
                f"\nDifference: {abs(orig_dist - new_dist):.10f}")
            return False
        
        orig_angle_val = orig_angles.get(pair_key, float('nan')) if orig_angles else float('nan')
        new_angle_val = new_angles.get(pair_key, float('nan')) if new_angles else float('nan')
        if abs(orig_angle_val - new_angle_val) > diff_tolerance:
            print(f"  Pair: {pair_key}")
            print(f"    Relative Angle Mismatch:    {orig_label} = {orig_angle_val:.2f} deg, {new_label} = {new_angle_val:.2f} deg"
                f"\nDifference: {abs(orig_angle_val - new_angle_val):.10f}")
            return False
    return True

def plot_relative_geometry(ax, cam_pos, cam_dir, numCam, cam_name, intersect):
    """
    Plots the relative geometry on a given Axes3D object.
    Does NOT create a new figure or call plt.show().
    """
    ax.set_title(str(cam_name))

    for i in range(numCam):
        ax.scatter(*cam_pos[i], s=100, label=f"{cam_name} {i+1} Pos")
        ax.quiver(*cam_pos[i], *cam_dir[i], length=100, color='blue', normalize=True)
    ax.scatter(*intersect, color='black', s=200, marker='X', label=f'{cam_name} Intersection')

    # Plot ground plane
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()

def save_transformed_mat(calib_dir:str, original_params_list:List[Dict[str, Any]], transformed_positions:NDArray, plane_rotation_matrix:NDArray):
    print("\nExporting transformed camera parameters to .mat files...")
    mat_save_folder = os.path.join(calib_dir, "Calibration")
    R_180_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    for i, orig_params in enumerate(original_params_list):
        # Combine all rotations
        #R_new_from_org = 
        R_final = orig_params['R_orig'] @ plane_rotation_matrix.T @ R_180_x
        P_final= transformed_positions[i]
        # New translation vector t = -R * C
        t_final = -R_final @ P_final.T

        mat_data = {
            'K': orig_params['K'],
            'RDistort': orig_params['RDistort'],
            'TDistort': orig_params['TDistort'],
            'r': R_final.T,
            't': t_final
        }
        
        mat_filename = f"hires_cam{i+1}_params.mat"
        mat_save_path = os.path.join(mat_save_folder, mat_filename)
        os.makedirs(mat_save_folder, exist_ok=True)
        sio.savemat(mat_save_path, mat_data)
        print(f"Saved: {mat_filename} for camera '{orig_params['name']}'")
    return mat_save_folder

def load_and_validate_mat(num_view:int, mat_save_folder:str):
    mat_file_list = []
    for i in range(num_view):
        mat_file_list.append(f"hires_cam{i+1}_params.mat")

    folder_files = os.listdir(mat_save_folder)

    # Check if ALL files in mat_file_list are present in files_in_folder
    if all(filename in folder_files for filename in mat_file_list):
        print("All .mat located, proceed to load and validate.")
    else:
        print("Missing .mat files. Saving process has probably failed.")
        missing_files = [filename for filename in mat_file_list if filename not in folder_files]
        print(f"Missing files: {missing_files}")
        return False
    
    mat_cam_positions = []
    mat_cam_orientations = []

    for matfile in mat_file_list:
        matfile_path = os.path.join(mat_save_folder, matfile)
        mat_data = sio.loadmat(matfile_path)
        r = mat_data['r'].T
        t = mat_data['t'].flatten()

        # Camera orientation
        cam_dir = r[2, :]
        mat_cam_orientations.append(cam_dir)

        # Calculate camera position from rotation and translation
        cam_pos = -np.dot(r.T, t)
        mat_cam_positions.append(cam_pos)

    mat_cam_positions = np.array(mat_cam_positions)
    mat_cam_orientations = np.array(mat_cam_orientations)

    return mat_cam_positions, mat_cam_orientations