import numpy as np
import open3d as o3d
import cv2
import os
from scipy.spatial.transform import Rotation

class GroundTruthFuser:
    def __init__(self, camera_matrix):
        # We use Open3D's native camera intrinsic object
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.intrinsic_matrix = camera_matrix
        
        self.global_pointcloud = o3d.geometry.PointCloud()

    def parse_trajectory_line(self, tx, ty, tz, qx, qy, qz, qw):
        """Converts x,y,z and quaternions into a 4x4 Pose Matrix"""
        transformation = np.eye(4)
        
        # Convert quaternion to 3x3 Rotation Matrix
        r = Rotation.from_quat([qx, qy, qz, qw])
        transformation[:3, :3] = r.as_matrix()
        
        # Insert translation (x, y, z)
        transformation[:3, 3] = [tx, ty, tz]
        
        return transformation

    def fuse_frame(self, rgb_path, depth_path, pose_matrix):
        """Projects raw TUM PNG depth into 3D using the perfect pose"""
        # 1. Load both RGB and Depth using Open3D's native image reader
        color_img = o3d.io.read_image(rgb_path)
        depth_img = o3d.io.read_image(depth_path)

        # 2. Create local RGB-D image 
        # IMPORTANT: TUM depth values are multiplied by 5000 to fit in a 16-bit PNG.
        # Setting depth_scale=5000.0 converts it back to perfect real-world meters.
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, 
            depth_img, 
            depth_scale=5000.0,  # CRITICAL FOR TUM DATASET
            depth_trunc=3.0,     # Ignore noise or background further than 3 meters
            convert_rgb_to_intensity=False
        )

        # 3. Generate Local Point Cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, 
            self.intrinsic
        )

        # 4. Move the local cloud to the perfect ground truth coordinate
        pcd.transform(pose_matrix)

        # 5. Add to the master map (Downsampling slightly to save RAM)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        self.global_pointcloud += pcd

    def save_map(self, filename="perfect_trajectory_map.ply"):
        print(f"Saving crisp, drift-free map to {filename}...")
        o3d.io.write_point_cloud(filename, self.global_pointcloud)


# --- Execution ---
if __name__ == "__main__":
    import glob

    # 1. TUM fr1 Intrinsic Matrix
    K = np.array([
        [517.3,   0.0, 318.6],
        [  0.0, 516.5, 255.3],
        [  0.0,   0.0,   1.0]
    ])

    fuser = GroundTruthFuser(camera_matrix=K)
    
    # 2. Paths to your raw TUM folders
    dataset_dir = "dataset_room" # Change this to your actual folder path containing rgb/, depth/, and groundtruth.txt
    trajectory_file = os.path.join(dataset_dir, "groundtruth.txt")
    
    if not os.path.exists(trajectory_file):
        print(f"Error: Could not find {trajectory_file}")
        exit()

    # 3. Load the entire ground truth trajectory into memory
    print("Loading ground truth trajectory database...")
    gt_poses = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            data = line.strip().split()
            # Store (timestamp, tx, ty, tz, qx, qy, qz, qw)
            gt_poses.append([float(x) for x in data])
    
    gt_poses = np.array(gt_poses)
    gt_timestamps = gt_poses[:, 0] # Extract just the times for fast lookup

    # 4. Find all actual PNG images inside your RGB folder
    rgb_images = sorted(glob.glob(os.path.join(dataset_dir, "rgb", "*.png")))
    print(f"Found {len(rgb_images)} actual image frames in the folder.")

    if len(rgb_images) == 0:
        print(f"Error: No images found in {os.path.join(dataset_dir, 'rgb')}. Check your folder path!")
        exit()

    print("Fusing frames based on closest timestamp matching...")
    fused_count = 0

    # 5. Loop through the actual images rather than the text file lines
    for rgb_path in rgb_images:
        # Extract the timestamp from the filename (e.g., "1305031102.175304")
        filename = os.path.basename(rgb_path)
        time_str = filename.replace(".png", "")
        img_time = float(time_str)

        # Find corresponding depth map with the exact same timestamp prefix
        depth_path = os.path.join(dataset_dir, "depth", filename)
        if not os.path.exists(depth_path):
            # Sometimes depth files have slightly different timestamps, let's find the closest one
            depth_matches = glob.glob(os.path.join(dataset_dir, "depth", f"{time_str[:-4]}*.png"))
            if depth_matches:
                depth_path = depth_matches[0]
            else:
                continue

        # THE FIX: Find the row in groundtruth.txt closest to this image's time
        time_diffs = np.abs(gt_timestamps - img_time)
        closest_idx = np.argmin(time_diffs)
        
        # If the closest tracking pose is more than 0.1 seconds away, ignore it (tracking lost)
        if time_diffs[closest_idx] > 0.1:
            continue

        # Extract the perfect coordinates from that matched row
        matched_pose = gt_poses[closest_idx]
        tx, ty, tz = matched_pose[1], matched_pose[2], matched_pose[3]
        qx, qy, qz, qw = matched_pose[4], matched_pose[5], matched_pose[6], matched_pose[7]

        # Build our 4x4 matrix
        perfect_pose = fuser.parse_trajectory_line(tx, ty, tz, qx, qy, qz, qw)

        # Fuse the data!
        # (Note: If using Depth Anything .npy arrays, change depth_path to point to your .npy files)
        fuser.fuse_frame(rgb_path, depth_path, perfect_pose)
        
        fused_count += 1
        if fused_count % 50 == 0:
            print(f"Successfully matched and fused {fused_count} frames...")

    # 6. Save the final cloud
    if fused_count > 0:
        fuser.save_map("tum_desk_perfect_sync.ply")
    else:
        print("Fatal Error: Could not match any images to the trajectory timestamps.")