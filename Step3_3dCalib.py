import os
import sleap_anipose as slap

rootpath = 'D:/Project/Sleap-Models/3dT/'

# Automatically picks the newest one
all_dirs = [d for d in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, d))]
project_dirs = [d for d in all_dirs if len(d) == 14 and d.isdigit()]
if not project_dirs:
    raise FileNotFoundError(f"No suitable project directories found in {rootpath}")
project_dirs.sort()
sessionpath = os.path.join(rootpath, project_dirs[-1])

cgroup, metadata = slap.calibrate(session = str(sessionpath), 
                                board = f"{sessionpath}/board.toml", 
                                excluded_views = (),
                                calib_fname = f"{sessionpath}/calibration.toml", 
                                metadata_fname = f"{sessionpath}/calibration.metadata.h5", 
                                histogram_path = f"{sessionpath}/reprojection_histogram.png", 
                                reproj_path = str(sessionpath))
