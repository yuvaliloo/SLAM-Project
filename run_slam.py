import open3d as o3d
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation

def load_ground_truth(filepath):
    """Reads the TUM groundtruth.txt and returns a dictionary of {timestamp: 4x4 Pose Matrix}"""
    print("Loading laboratory ground truth poses...")
    gt_poses = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            
            parts = line.strip().split()
            if len(parts) != 8: continue
            
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # Convert Translation and Quaternion into a 4x4 matrix
            pose = np.identity(4)
            pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            pose[:3, 3] = [tx, ty, tz]
            
            gt_poses[timestamp] = pose
            
    return gt_poses

def run_groundtruth_slam():
    # 1. Exact Physical Lens Math
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480, fx=517.3, fy=516.5, cx=318.6, cy=255.3
    )

    # 2. Get the ground truth poses
    # MAKE SURE THIS PATH IS CORRECT
    gt_poses = load_ground_truth("dataset_room/groundtruth.txt")

    # 3. Grab the raw files
    rgb_files = sorted(glob.glob("dataset_room/rgb/*.png"))
    depth_files = sorted(glob.glob("dataset_room/depth/*.png"))

    def extract_time(filepath):
        name_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        try: return float(name_no_ext)
        except: return 0.0

    print("Building perfect 3D map using Ground Truth...")

    # TSDF Volume to absorb minor sensor fuzz
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.015,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    fused_count = 0

    for i in range(min(len(rgb_files), len(depth_files))):
        rgb_time = extract_time(rgb_files[i])
        
        # Find the absolute closest ground truth pose for this image
        if not gt_poses: break
        closest_gt_time = min(gt_poses.keys(), key=lambda k: abs(k - rgb_time))
        
        # If the ground truth is too far away in time, skip it to avoid blurring
        if abs(closest_gt_time - rgb_time) > 0.05:
            continue

        perfect_pose = gt_poses[closest_gt_time]

        color = o3d.io.read_image(rgb_files[i])
        depth = o3d.io.read_image(depth_files[i])
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=5000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        # Print the frame exactly where the motion-capture cameras said it was
        extrinsic = np.linalg.inv(perfect_pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
        fused_count += 1
        print(f"Fused frame {i} using Ground Truth.")

    print(f"--- COMPLETE. Fused {fused_count} perfectly tracked frames. ---")
    
    # Extract and render
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # Save it so you don't have to run it again
    o3d.io.write_triangle_mesh("Perfect_Deliverable.ply", mesh)
    
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    run_groundtruth_slam()