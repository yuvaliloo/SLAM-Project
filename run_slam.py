import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import glob
import os 

class VisualDepthSLAMNoOpen3D:
    def __init__(self, camera_matrix=None):
        print("Loading Depth Anything V2 (Metric Indoor)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # FIXED: Both processor and model must point to the Metric-Indoor checkpoint
        model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.depth_model.eval()
        
        # Tracking Frontend Setup - Upgraded for aggressive indoor texture extraction
        self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Camera Intrinsic Matrix (Placeholder - adjust to your specific iPhone/camera if known)
        if camera_matrix is not None:
            self.K = camera_matrix
        else:
            self.K = np.array([[525.0,   0.0, 320.0],
                               [  0.0, 525.0, 240.0],
                               [  0.0,   0.0,   1.0]], dtype=np.float32)
            
        self.current_pose = np.eye(4)
        self.prev_frame = None
        self.prev_kps = None
        self.prev_des = None
        self.prev_depth = None

        # Storage for global map points
        self.global_points = []
        self.global_colors = []

    @torch.no_grad()
    def get_dense_depth(self, frame):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.depth_model(**inputs)
        
        post_processed_output = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(frame.shape[0], frame.shape[1])]
        )
        depth_map = post_processed_output[0]["predicted_depth"].cpu().numpy()
        
        # Note: True metric models do not require the relative inverse calculation inversion layer 
        return depth_map

    def track_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to ensure textures pop out
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        kps, des = self.orb.detectAndCompute(gray, None)
        
        # Handle the very first frame initialization
        if self.prev_frame is None or self.prev_depth is None:
            self.prev_frame = gray
            self.prev_kps = kps
            self.prev_des = des
            return self.current_pose, False

        # SAFEGUARD: If current frame has no features, update baseline anyway and skip math
        if des is None or len(kps) < 15 or self.prev_des is None:
            self.prev_frame = gray
            self.prev_kps = kps
            self.prev_des = des
            return self.current_pose, False

        # Match features between previous frame (t-1) and current frame (t)
        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:120]

        # If matching quality is poor, update baseline and skip pose calculation
        if len(matches) < 12:
            self.prev_frame = gray
            self.prev_kps = kps
            self.prev_des = des
            return self.current_pose, False

        pts_3d = []
        pts_2d = []
        
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        for m in matches:
            u, v = int(self.prev_kps[m.queryIdx].pt[0]), int(self.prev_kps[m.queryIdx].pt[1])
            z = self.prev_depth[v, u] 
            
            if z < 0.2 or z > 8.0: 
                continue
                
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            pts_3d.append([x, y, z])
            pts_2d.append(kps[m.trainIdx].pt)

        # Update the baseline tracking states BEFORE executing early exits
        # This prevents the system from getting stuck in a historical feedback loop
        self.prev_frame = gray
        self.prev_kps = kps
        self.prev_des = des

        if len(pts_3d) < 10:
            return self.current_pose, False

        pts_3d = np.float32(pts_3d)
        pts_2d = np.float32(pts_2d)

        # Solve PnP for camera delta position
        success, rvec, t, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, self.K, None, 
                                                       reprojectionError=4.0, confidence=0.99)
        
        if not success or inliers is None or len(inliers) < 8:
            return self.current_pose, False
            
        R, _ = cv2.Rodrigues(rvec)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()

        # Accumulate the relative motion transformation
        self.current_pose = self.current_pose @ np.linalg.inv(T)
        
        return self.current_pose, True

    def accumulate_3d_points(self, rgb_image, depth_map, pose):
        h, w = depth_map.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Optimized threshold for typical indoor room geometries (0.2m to 8.0m max)
        valid_mask = (depth_map > 0.2) & (depth_map < 8.0)
        
        # Subsample points (take every 6th pixel for crisp indoor details without blowing up file size)
        subsample = np.zeros_like(valid_mask, dtype=bool)
        subsample[::6, ::6] = True
        valid_mask = valid_mask & subsample

        z = depth_map[valid_mask]
        u = u[valid_mask]
        v = v[valid_mask]
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        pts_local = np.vstack((x, y, z, np.ones_like(z)))
        pts_global = (pose @ pts_local)[:3, :].T
        colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)[valid_mask]

        self.global_points.append(pts_global)
        self.global_colors.append(colors)

    def save_to_ply(self, filename="indoor_slam_map.ply"):
        if not self.global_points:
            print("No 3D points accumulated.")
            return

        all_pts = np.vstack(self.global_points)
        all_cols = np.vstack(self.global_colors)

        print(f"Writing {len(all_pts)} indoor cloud points to {filename}...")
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(all_pts, all_cols):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        print("File saved successfully!")
def get_time(filepath):
        basename = os.path.basename(filepath)
        return float(basename.replace('.png', ''))
# --- Runtime Loop Execution ---
if __name__ == "__main__":
    # Ensure this matrix matches the TUM fr1 specs!
    tum_fr1_matrix = np.array([
        [517.3,   0.0, 318.6],
        [  0.0, 516.5, 255.3],
        [  0.0,   0.0,   1.0]
    ], dtype=np.float32)

    slam = VisualDepthSLAMNoOpen3D(camera_matrix=tum_fr1_matrix)
    
    # 1. Load both sets of files
    rgb_files = sorted(glob.glob("rgb_room/*.png"))
    depth_files = sorted(glob.glob("depth_room/*.png"))
    
    if not rgb_files or not depth_files:
        print("Error: Missing rgb or depth folders!")
        exit()

    print(f"Validating SLAM against {len(rgb_files)} ground-truth frames...")
    
    # Pre-extract depth timestamps for fast nearest-neighbor matching
    depth_times = np.array([get_time(f) for f in depth_files])

    for img_path in rgb_files:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
            
        # 2. Synchronize: Find the closest physical depth map to this RGB frame
        rgb_time = get_time(img_path)
        closest_depth_idx = np.abs(depth_times - rgb_time).argmin()
        closest_depth_path = depth_files[closest_depth_idx]

        # 3. Load Hardware Depth (IMREAD_ANYDEPTH is critical for 16-bit PNGs)
        raw_depth = cv2.imread(closest_depth_path, cv2.IMREAD_ANYDEPTH)
        
        # 4. Convert TUM 16-bit format to Metric Meters
        # 0 values are hardware sensor failures (too close/too far/shiny surface)
        depth_map = raw_depth.astype(np.float32) / 5000.0 
        
        # We NO LONGER call slam.get_dense_depth(frame). 
        # The AI is completely bypassed.
        
        # Resize RGB to match the 640x480 hardware depth map
        frame = cv2.resize(frame, (640, 480))
        
        # 5. Run the SLAM pose estimation
        pose, success = slam.track_features(frame)
        
        debug_frame = frame.copy()
        if success:
            slam.accumulate_3d_points(frame, depth_map, pose)
            cv2.putText(debug_frame, "Hardware Mapping Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
        else:
            cv2.putText(debug_frame, "Tracking Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if slam.prev_kps is not None:
            debug_frame = cv2.drawKeypoints(debug_frame, slam.prev_kps, None, color=(0, 255, 0), flags=0)

        slam.prev_depth = depth_map

        cv2.imshow("Ground Truth Validator", debug_frame)
        if cv2.waitKey(30) & 0xFF == 27: 
            break

    cv2.destroyAllWindows()
    slam.save_to_ply("hardware_validation_map.ply")