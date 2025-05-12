import os
import tomllib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.spatial.transform as st
from itertools import combinations

# --- Project parameters, adjust this to where your calibration.toml is stored!
projectDir = 'D:/Project/Sleap-Models/3dT/20250504191009'

# --- Helper functions for relative geometry ---
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
        dir2 = orientations[j] / np.linalg.norm(orientations[j]) # Ensure normalized
        dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0) # Clip for numerical stability
        angle_rad = np.arccos(dot_product)
        angles[pair_key] = np.degrees(angle_rad)
    return angles

def print_relative_geometry_stats(positions, orientations, label=""):
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
    print(f"\n--- Comparison: {orig_label} vs. {new_label} Relative Geometry ---")
    if orig_distances is None or not orig_distances:
        print(f"  Cannot perform comparison: Missing or empty {orig_label} data.")
        return
    if new_distances is None or not new_distances:
        print(f"  Cannot perform comparison: Missing or empty {new_label} data.")
        return

    all_pair_keys = sorted(list(set(orig_distances.keys()) | set(new_distances.keys())))

    for pair_key in all_pair_keys:
        print(f"  Pair: {pair_key}")
        
        orig_dist = orig_distances.get(pair_key, float('nan'))
        new_dist = new_distances.get(pair_key, float('nan'))
        print(f"    Relative Distance: {orig_label} = {orig_dist:.2f}, {new_label} = {new_dist:.2f}")
        
        orig_angle_val = orig_angles.get(pair_key, float('nan')) if orig_angles else float('nan')
        new_angle_val = new_angles.get(pair_key, float('nan')) if new_angles else float('nan')
        print(f"    Relative Angle:    {orig_label} = {orig_angle_val:.2f} deg, {new_label} = {new_angle_val:.2f} deg")

sleapCalib = os.path.join(projectDir,'calibration.toml')
camera_names = []

try:
    with open(sleapCalib, 'rb') as f:
        sCal = tomllib.load(f)
        camNO = 0
        camera_positions = []
        original_cam_dirs_list = []
        original_params_list = [] # To store K, R_orig, RDistort, TDistort
        # Prepare for plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Calibration Visualization')

    for camera_name, camera_data in sCal.items():
        if 'matrix' in camera_data:

            camNO = camNO + 1
            # Original parameters
            K_orig = np.array(camera_data['matrix']).T
            rotations_orig = np.array(camera_data['rotation'])
            # Convert rotation vector to rotation matrix
            R_orig = st.Rotation.from_rotvec(rotations_orig).as_matrix()
            t_orig = np.array(camera_data['translation']) # Original translation vector
            RDistort_orig = np.array(camera_data['distortions'][0:2])
            TDistort_orig = np.array(camera_data['distortions'][2:4])

            original_params_list.append({
                'K': K_orig, 'R_orig': R_orig, 'RDistort': RDistort_orig, 'TDistort': TDistort_orig,
                'name': camera_name # Store name for potential use in filename if needed
            })

            # Camera position (negate translation because it represents the world origin in camera coordinates)
            cam_pos = -np.dot(R_orig.T, t_orig)

            # Camera orientation (principal axis)
            cam_dir = R_orig[2, :]  # The third row of the rotation matrix represents the camera's principal axis
            camera_positions.append(cam_pos)
            original_cam_dirs_list.append(cam_dir)

            # Plot camera position
            #ax.scatter(*cam_pos, label=f'Camera {camNO} Position')
            ax.scatter(*cam_pos, label=f'Camera {camNO} Position', s=100)

            # Customize quiver appearance (e.g., length, color)
            camera_names.append(camera_name)
            ax.quiver(*cam_pos, *cam_dir, length=100, normalize=True, label=f'Camera {camNO} Orientation', color=plt.cm.tab10(camNO-1)) # Assign different color to each camera

    camera_positions = np.array(camera_positions)
    original_cam_dirs = np.array(original_cam_dirs_list)

    # --- Calculate and print original relative geometry ---
    original_rel_distances, original_rel_angles = print_relative_geometry_stats(
        camera_positions, original_cam_dirs, "Original Setup"
    )
    # Calculate the intersection point of the original camera orientations
    num_cameras_orig = len(camera_positions)
    A_lstsq = np.zeros((3 * num_cameras_orig, 3))
    b_lstsq = np.zeros(3 * num_cameras_orig)
    I_3x3 = np.eye(3)

    for i in range(num_cameras_orig):
        Ci = camera_positions[i]
        di = original_cam_dirs[i] # Assumed normalized from rotation matrix
        
        # Projection matrix onto the plane orthogonal to di
        Mi = I_3x3 - np.outer(di, di)
        
        A_lstsq[3*i : 3*(i+1), :] = Mi
        b_lstsq[3*i : 3*(i+1)] = Mi @ Ci

    P_intersect_orig = np.linalg.lstsq(A_lstsq, b_lstsq, rcond=None)[0]
    ax.scatter(*P_intersect_orig, color='black', s=200, marker='X', label='Original Intersection Point', depthshade=True, zorder=10)

    # Plane fitting (PCA)
    centroid = np.mean(camera_positions, axis=0)
    centered_positions = camera_positions - centroid
    covariance_matrix = np.cov(centered_positions, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    # Ensure the normal vector points "upward" (positive z-component)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector
    
    # Define ground plane normal
    ground_normal = np.array([0, 0, 1])

    # Calculate rotation to align planes
    rotation_vector = np.cross(normal_vector, ground_normal)
    rotation_vector_norm = np.linalg.norm(rotation_vector)
    if rotation_vector_norm > 1e-6: # Avoid division by zero
        rotation_axis = rotation_vector / rotation_vector_norm
        rotation_angle = np.arccos(np.dot(normal_vector, ground_normal))
        rotation = st.Rotation.from_rotvec(rotation_axis * rotation_angle)
        rotation_matrix = rotation.as_matrix()
    else:
        rotation_matrix = np.eye(3) # No rotation needed if planes are already aligned.

    # --- Adjust Z-level based on the intersection point ---
    # Rotate the original intersection point around the centroid
    rotated_centered_P_intersect = np.dot(rotation_matrix, (P_intersect_orig - centroid))

    # Define target Z for the intersection point after transformation
    target_intersection_z = 0.0

    # Calculate the Z translation needed for the entire system
    z_translation_for_system = target_intersection_z - rotated_centered_P_intersect[2]

    # Apply rotation to positions and orientations
    # These are rotated camera positions, centered at the origin (because 'centroid' was subtracted)
    transformed_positions = np.dot(rotation_matrix, (camera_positions - centroid).T).T 
    # Now apply the Z translation to make the transformed intersection point land on Z=target_intersection_z
    transformed_positions[:, 2] += z_translation_for_system

    # The final transformed intersection point will be:
    final_P_intersect = rotated_centered_P_intersect + np.array([0, 0, z_translation_for_system])
    ax.scatter(*final_P_intersect, color='purple', s=200, marker='P', label=f'Transformed Intersection (Z={target_intersection_z:.1f})', depthshade=True, zorder=10)

    # --- Apply 180-degree rotation if cameras are mostly in negative Z ---
    # (or always, to ensure they are in positive Z relative to intersection)
    # We will rotate around the X-axis passing through final_P_intersect.
    # This keeps final_P_intersect at Z=0 but flips other Zs relative to it.
    
    # Rotation matrix for 180 degrees around X-axis
    R_180_x = np.array([[1,  0,  0],
                         [0, -1,  0],
                         [0,  0, -1]])

    # Apply this rotation to camera positions (relative to final_P_intersect)
    temp_transformed_positions = np.zeros_like(transformed_positions)
    for i in range(len(transformed_positions)):
        pos_relative_to_intersect = transformed_positions[i] - final_P_intersect
        rotated_pos_relative = R_180_x @ pos_relative_to_intersect
        temp_transformed_positions[i] = rotated_pos_relative + final_P_intersect
    transformed_positions = temp_transformed_positions

    transformed_orientations = []
    plot_cam_idx = 0 # Use a 0-based index for plotting transformed cameras
    for camera_name, camera_data in sCal.items():
        if 'matrix' in camera_data:
            # Use the original_cam_dirs collected earlier, transformed
            original_dir_for_this_cam = original_cam_dirs[plot_cam_idx]
            # First, apply the plane alignment rotation
            intermediate_dir = np.dot(rotation_matrix, original_dir_for_this_cam)
            # Then, apply the 180-degree flip rotation
            transformed_dir = R_180_x @ intermediate_dir
            transformed_orientations.append(transformed_dir)
            ax.quiver(*transformed_positions[plot_cam_idx], *transformed_dir, length=100, normalize=True, alpha=0.8, color=plt.cm.Set1(plot_cam_idx), linewidths=2, label=f'Camera {plot_cam_idx+1} Orientation (Transformed)')
            ax.scatter(*transformed_positions[plot_cam_idx], alpha=0.5, color=plt.cm.Set1(plot_cam_idx), s=150, linewidths=2, label=f'Camera {plot_cam_idx+1} Position (Transformed)')
            # Add label to transformed camera position
            ax.text(transformed_positions[plot_cam_idx][0], transformed_positions[plot_cam_idx][1], transformed_positions[plot_cam_idx][2], f"{camera_names[plot_cam_idx]} (T)", fontsize=10, color=plt.cm.Set1(plot_cam_idx))
            plot_cam_idx +=1

    transformed_orientations_np = np.array(transformed_orientations)

    # --- Calculate, print, and compare transformed (in-memory) relative geometry ---
    transformed_rel_distances_mem, transformed_rel_angles_mem = print_relative_geometry_stats(
        transformed_positions, transformed_orientations_np, "Transformed Setup (In-Memory)"
    )
    if original_rel_distances and transformed_rel_distances_mem:
        compare_relative_geometries(original_rel_distances, original_rel_angles,
                                    transformed_rel_distances_mem, transformed_rel_angles_mem,
                                    new_label="Transformed (In-Memory)")
    # Define ground plane for visualization
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    X, Y = np.meshgrid(np.arange(xlim[0], xlim[1], 10), np.arange(ylim[0], ylim[1], 10))
    Z = np.full_like(X, target_intersection_z) # Ground plane at the target Z of intersection

    # Plot ground plane
    ax.plot_surface(X, Y, Z, alpha=0.2, color='gray', label='Ground Plane')

    # Set plot limits
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_zlim([-500, 500])

    # Adjust legend to show all elements
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate labels
    #ax.legend(by_label.values(), by_label.keys(), loc='')
    
    # Add labels near each camera position
    for i, pos in enumerate(camera_positions):
        ax.text(pos[0], pos[1], pos[2], f"{camera_names[i]}", fontsize=12)

    # --- Export Transformed Parameters to .mat files ---
    print("\nExporting transformed camera parameters to .mat files...")
    for idx in range(len(original_params_list)):
        orig_cam_params = original_params_list[idx]
        
        K_to_save = orig_cam_params['K']
        R_orig_for_cam = orig_cam_params['R_orig']
        RDistort_to_save = orig_cam_params['RDistort']
        TDistort_to_save = orig_cam_params['TDistort']

        # Final rotation matrix for this camera
        R_final_cam = R_orig_for_cam @ rotation_matrix.T @ R_180_x
        
        # Final position of this camera in world coordinates
        P_final_cam = transformed_positions[idx]
        
        # Final translation vector t_wc = -R_wc * C_w
        t_final_cam = -R_final_cam @ P_final_cam.T
        
        Saving2Mat_transformed = {
            'K': K_to_save, 'r': R_final_cam.T, 'RDistort': RDistort_to_save,
            't': t_final_cam, 'TDistort': TDistort_to_save
        }
        mat_filename_transformed = f'hires_cam{idx+1}_params.mat' # Using 1-based indexing for filename
        mat_save_path = os.path.join(projectDir, mat_filename_transformed)
        sio.savemat(mat_save_path, Saving2Mat_transformed)
        print(f"Transformed data for camera {idx+1} ({orig_cam_params['name']}) saved to {mat_filename_transformed}")

    # --- Validation: Load and plot the generated .mat files ---
    print("\nValidation: Loading and plotting generated .mat files...")

    fig_val = plt.figure()
    ax_val = fig_val.add_subplot(111, projection='3d')
    ax_val.set_box_aspect([1, 1, 1])
    ax_val.set_xlabel('X')
    ax_val.set_ylabel('Y')
    ax_val.set_zlabel('Z')
    ax_val.set_title('Validation: Camera Positions from .mat Files')
    ax_val.set_xlim([-500, 500])
    ax_val.set_ylim([-500, 500])
    ax_val.set_zlim([-500, 500])

    mat_cam_positions_val = []
    mat_cam_orientations_val = []

    for i in range(1, camNO + 1):  # Assuming filenames are hires_cam1_params.mat, ...
        mat_filename = f'hires_cam{i}_params.mat'
        mat_load_path = os.path.join(projectDir, mat_filename)
        try:
            mat_data = sio.loadmat(mat_load_path)
            K = mat_data['K']
            r = mat_data['r']
            t = mat_data['t']

            # Calculate camera position from rotation and translation
            cam_pos = -np.dot(r, t.flatten()) # Flatten t to 1D array
            mat_cam_positions_val.append(cam_pos)
            
            # Camera orientation
            cam_dir = r.T[2, :]
            
            # Plot the camera position and orientation
            ax_val.scatter(*cam_pos, label=f'Camera {i} Position', s=100)
            ax_val.quiver(*cam_pos, *cam_dir, length=100, normalize=True, label=f'Camera {i} Orientation', color=plt.cm.tab10(i-1))
            mat_cam_orientations_val.append(cam_dir)
            
            print(f"Loaded and plotted data from {mat_filename}")

        except FileNotFoundError:
            print(f"Error: Could not find {mat_filename} at {mat_load_path}")
        except Exception as e:
            print(f"Error loading or plotting {mat_filename}: {e}")

    mat_cam_positions_val = np.array(mat_cam_positions_val)
    mat_cam_orientations_val = np.array(mat_cam_orientations_val)

    # --- Calculate, print, and compare relative geometry from .mat files ---
    # This compares the original setup to the setup loaded from the final .mat files
    mat_rel_distances, mat_rel_angles = print_relative_geometry_stats(
        mat_cam_positions_val, mat_cam_orientations_val, "Setup from .mat Files"
    )
    if original_rel_distances and mat_rel_distances:
        compare_relative_geometries(original_rel_distances, original_rel_angles,
                                    mat_rel_distances, mat_rel_angles,
                                    new_label="From .mat Files")
    ax_val.legend() 
    
except FileNotFoundError:
    print(f"Error: File not found at {sleapCalib}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

plt.show()
