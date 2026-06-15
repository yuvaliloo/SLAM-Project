import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

class VisualDepthSLAMNoOpen3D:
    def __init__(self, camera_matrix=None):
        print("Loading Depth Anything V2...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(self.device)
        self.depth_model.eval()
        
        # Tracking Frontend Setup
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
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
        depth_map = 1.0 / (depth_map + 1e-5)
        return depth_map

    def track_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kps = kps
            self.prev_des = des
            return self.current_pose, False

        matches = self.bf.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) < 10:
            return self.current_pose, False

        pts_prev = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kps[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(pts_curr, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return self.current_pose, False
            
        _, R, t, mask_pose = cv2.recoverPose(E, pts_curr, pts_prev, self.K, mask=mask)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()

        self.current_pose = self.current_pose @ np.linalg.inv(T)
        
        self.prev_frame = gray
        self.prev_kps = kps
        self.prev_des = des
        return self.current_pose, True

    def accumulate_3d_points(self, rgb_image, depth_map, pose):
        """Reprojects pixels and accumulates them into a raw list."""
        h, w = depth_map.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid_mask = (depth_map > 0.1) & (depth_map < 10.0)
        
        # Subsample points (take every 8th pixel) to prevent memory issues and huge files
        subsample = np.zeros_like(valid_mask, dtype=bool)
        subsample[::8, ::8] = True
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

    def save_to_ply(self, filename="slam_output.ply"):
        """Saves all accumulated points to a standard 3D PLY file."""
        if not self.global_points:
            print("No 3D points accumulated.")
            return

        all_pts = np.vstack(self.global_points)
        all_cols = np.vstack(self.global_colors)

        print(f"Writing {len(all_pts)} points to {filename}...")
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

# --- Runtime Loop ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0) # Uses your webcam, or swap with a video path string
    slam = VisualDepthSLAMNoOpen3D()

    print("Running tracking loop. Press 'ESC' to stop and export the 3D Map.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        pose, success = slam.track_features(frame)
        depth_map = slam.get_dense_depth(frame)
        
        if success:
            slam.accumulate_3d_points(frame, depth_map, pose)
            cv2.putText(frame, "Mapping Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("SLAM Video Feed Tracker", frame)
        
        # Break Loop with Escape Key (Keycode 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Export the final map data
    slam.save_to_ply("drone_slam_map.ply")