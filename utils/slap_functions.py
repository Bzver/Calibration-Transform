import os

import sleap_anipose as slap

def generate_calib_board(calib_dir:str):
    board_pic = os.path.join(calib_dir, "board.jpg")
    board_toml = os.path.join(calib_dir, "board.toml")
    if os.path.exists(board_pic) and os.path.exists(board_toml):
        print("Existing board.png and board.toml detected. Skipping...")
        return
    print(f"Generating calibration board and configuration at {calib_dir}...")
    slap.draw_board(
        board_name = board_pic, board_x = 8, board_y = 11, 
        square_length = 24.0, marker_length = 19.75, marker_bits = 4, 
        dict_size = 1000, img_width = 1440, img_height = 1440, save = board_toml
        )
    
def calibration(calib_dir:str):
    try:
        cgroup, metadata = slap.calibrate(
            session = str(calib_dir),   board = f"{calib_dir}/board.toml",  excluded_views = (),
            calib_fname = f"{calib_dir}/calibration.toml",  metadata_fname = f"{calib_dir}/calibration.metadata.h5", 
            histogram_path = f"{calib_dir}/reprojection_histogram.png", reproj_path = str(calib_dir)
            )
        return True
    except Exception as e:
        print(f"Sleap-Anipose calibration failed: {e}")
        return False