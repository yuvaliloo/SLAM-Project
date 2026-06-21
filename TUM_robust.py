import open3d as o3d
import numpy as np
import glob
import os

def get_synced_dataset(rgb_dir, depth_dir, max_time_diff=0.02):
    rgb_files = glob.glob(f"{rgb_dir}/*.*")
    depth_files = glob.glob(f"{depth_dir}/*.png")
    
    def extract_time(f):
        try: return float(os.path.splitext(os.path.basename(f))[0])
        except: return 0.0
        
    rgb_dict = {extract_time(f): f for f in rgb_files if f.endswith(('.png', '.jpg'))}
    depth_dict = {extract_time(f): f for f in depth_files if f.endswith('.png')}
    
    synced_rgb, synced_depth = [], []
    for rgb_time in sorted(rgb_dict.keys()):
        if not depth_dict: break
        closest_depth_time = min(depth_dict.keys(), key=lambda k: abs(k - rgb_time))
        if abs(closest_depth_time - rgb_time) < max_time_diff:
            synced_rgb.append(rgb_dict[rgb_time])
            synced_depth.append(depth_dict[closest_depth_time])
    return synced_rgb, synced_depth

def run_tum_robust_slam():
    # 1. Exact Dual-Lens Intrinsics for TUM FR1
    rgb_intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3)
    # The depth lens is slightly 'zoomed in' compared to the RGB lens
    depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 591.1, 590.1, 331.0, 234.0)

    rgb_files, depth_files = get_synced_dataset("dataset_room/rgb", "dataset_room/depth")
    rgb_files, depth_files = rgb_files, depth_files

    print("Executing TUM Robust Odometry Math...")

    # --- THE TUM MATH: Robust Penalty Function ---
    # The Huber Loss function acts as a mathematical guillotine. 
    # Any pixel deviation larger than 5cm (0.05m) is classified as motion blur or occlusion and is ignored.
    robust_kernel = o3d.pipelines.registration.HuberLoss(k=0.05)
    
    # We apply this kernel to Point-to-Plane ICP to force walls to stay flat
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(robust_kernel)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.008, sdf_trunc=0.025, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    current_pose = np.identity(4)
    prev_pcd = None

    for i in range(len(rgb_files)):
        color = o3d.io.read_image(rgb_files[i])
        depth = o3d.io.read_image(depth_files[i])
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=5000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        # Generate the Point Cloud using the correct, separated depth math
        curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, depth_intrinsic)
        
        # Calculate surface angles for the Point-to-Plane math
        curr_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        if i == 0:
            prev_pcd = curr_pcd
            extrinsic = np.linalg.inv(current_pose)
            volume.integrate(rgbd, rgb_intrinsic, extrinsic)
            continue

        # --- THE CALCULATION ---
        # Instead of generic RGB-D odometry, we force the robust, kernel-backed ICP loop
        icp_result = o3d.pipelines.registration.registration_icp(
            curr_pcd, prev_pcd, 
            max_correspondence_distance=0.05, # Search radius
            init=np.identity(4),              # Assume small movement
            estimation_method=estimation_method
        )

        # Only accept the math if the fitness (overlap) is high enough
        if icp_result.fitness > 0.3:
            # Apply the mathematically robust rotation/translation
            current_pose = np.dot(current_pose, icp_result.transformation)
            prev_pcd = curr_pcd
            
            # Print to map every 3 frames
            if i % 5 == 0:
                extrinsic = np.linalg.inv(current_pose)
                volume.integrate(rgbd, rgb_intrinsic, extrinsic)
                print(f"Robust lock secured. Fused frame {i}")
        else:
            print(f"Frame {i} rejected by Robust Kernel (Motion Blur).")

    print("--- SLAM COMPLETE. RUNNING POST-PROCESSING FILTER ---")
    
    # 1. Extract the raw points from the TSDF sponge
    raw_pcd = volume.extract_point_cloud()
    print(f"Raw points: {len(raw_pcd.points)}")

    # 2. Statistical Outlier Removal (The Vacuum Cleaner)
    # nb_neighbors: How many nearby points to check
    # std_ratio: How aggressive the filter is (lower = more aggressive deletion of fuzz)
    print("Filtering hardware noise and flying pixels...")
    clean_pcd, ind = raw_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    
    # 3. Recalculate 3D lighting angles so the surfaces pop
    clean_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    print(f"Clean points: {len(clean_pcd.points)}")
    print("Opening final presentation viewer...")
    
    # Render the cleaned, razor-sharp point cloud
    o3d.visualization.draw_geometries([clean_pcd])

if __name__ == "__main__":
    run_tum_robust_slam()