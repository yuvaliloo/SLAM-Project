import numpy as np
import open3d as o3d
import cv2
import os
import glob
import torch
from scipy.spatial.transform import Rotation
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image as PILImage

class GroundTruthICPFuser:
    def __init__(self, camera_matrix):
        # 1. Open3D Intrinsic Setup
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.intrinsic_matrix = camera_matrix
        self.global_pointcloud = o3d.geometry.PointCloud()
        
        # 2. In-Memory AI Initialization
        print("Loading Depth Anything V2 onto GPU...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.depth_model.eval()

    def parse_trajectory_line(self, tx, ty, tz, qx, qy, qz, qw):
        """Converts x,y,z and quaternions into a 4x4 Pose Matrix"""
        transformation = np.eye(4)
        r = Rotation.from_quat([qx, qy, qz, qw])
        transformation[:3, :3] = r.as_matrix()
        transformation[:3, 3] = [tx, ty, tz]
        return transformation

    def fuse_frame(self, rgb_path, pose_matrix):
        """Runs AI depth, applies scale correction, and dynamically aligns using ICP"""
        # 1. Image Prep
        frame_bgr = cv2.imread(rgb_path)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        
        # 2. AI Inference
        image_pil = PILImage.fromarray(frame_rgb)
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
        
        depth_map = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(h, w)]
        )[0]["predicted_depth"].cpu().numpy()

        # 3. BASELINE SCALE CORRECTION
        # Get the AI roughly to the physical scale so ICP has a chance to lock on.
        # (Tune this if the map is still separating vastly!)
        depth_map = depth_map * 0.85 

        # 4. Create Open3D structures
        color_img = o3d.geometry.Image(frame_rgb)
        depth_img = o3d.geometry.Image(depth_map)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, 
            depth_scale=1.0, depth_trunc=3.5, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)

        # 5. THE MAGIC: ICP ALIGNMENT
        if self.global_pointcloud.is_empty():
            # First frame ever: just place it where the trajectory says
            pcd.transform(pose_matrix)
            self.global_pointcloud += pcd
        else:
            # We downsample temporarily just for the math so ICP runs fast
            source_down = pcd.voxel_down_sample(voxel_size=0.05)
            target_down = self.global_pointcloud.voxel_down_sample(voxel_size=0.05)
            
            # Max distance ICP is allowed to look for a matching wall (15 centimeters)
            threshold = 0.15 
            
            # Run Iterative Closest Point using Ground Truth as the initial guess
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold, pose_matrix,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            # Apply the ICP's mathematically refined matrix instead of the raw trajectory
            pcd.transform(reg_p2p.transformation)
            
            # Merge into master map
            self.global_pointcloud += pcd

        # Clean up memory
        self.global_pointcloud = self.global_pointcloud.voxel_down_sample(voxel_size=0.01)


    def save_map(self, filename="ai_icp_reconstruction.ply"):
        print(f"Saving crisp, drift-free map to {filename}...")
        o3d.io.write_point_cloud(filename, self.global_pointcloud)


# --- Execution ---
if __name__ == "__main__":
    # TUM Intrinsic Matrix
    K = np.array([
        [517.3,   0.0, 318.6],
        [  0.0, 516.5, 255.3],
        [  0.0,   0.0,   1.0]
    ])

    fuser = GroundTruthICPFuser(camera_matrix=K)
    
    dataset_dir = "dataset_room"  # Update path if needed
    trajectory_file = os.path.join(dataset_dir, "groundtruth.txt")
    
    print("Loading trajectory...")
    gt_poses = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "": continue
            gt_poses.append([float(x) for x in line.strip().split()])
    
    gt_poses = np.array(gt_poses)
    gt_timestamps = gt_poses[:, 0]

    rgb_images = sorted(glob.glob(os.path.join(dataset_dir, "rgb", "*.png")))
    print(f"Processing {len(rgb_images)} frames with ICP correction...")

    fused_count = 0
    # Still skipping 3 frames to save time, ICP handles larger jumps perfectly
    for rgb_path in rgb_images[::3]:
        img_time = float(os.path.basename(rgb_path).replace(".png", ""))
        
        time_diffs = np.abs(gt_timestamps - img_time)
        closest_idx = np.argmin(time_diffs)
        if time_diffs[closest_idx] > 0.1: continue

        matched_pose = gt_poses[closest_idx]
        tx, ty, tz = matched_pose[1], matched_pose[2], matched_pose[3]
        qx, qy, qz, qw = matched_pose[4], matched_pose[5], matched_pose[6], matched_pose[7]

        perfect_pose = fuser.parse_trajectory_line(tx, ty, tz, qx, qy, qz, qw)

        # Execute ICP Fusion
        fuser.fuse_frame(rgb_path, perfect_pose)
        
        fused_count += 1
        if fused_count % 10 == 0:
            print(f"Fused {fused_count} frames utilizing ICP micro-adjustments...")

    if fused_count > 0:
        fuser.save_map("tum_room_ai_icp.ply")