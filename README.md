# Robust RGB-D SLAM (TUM Architecture)

## 💡 Project Idea
While modern zero-shot AI depth models (like Depth Anything) produce visually stunning relative depth maps, they fundamentally lack the strict metric scale required for dense 3D geometric tracking. Attempting to run Iterative Closest Point (ICP) algorithms on AI-hallucinated depth results in catastrophic scale drift and geometric warping.

This project implements a robust, physics-based Offline SLAM Architecture designed specifically to handle raw, noisy hardware sensor data (like the Microsoft Kinect v1 used in the TUM dataset). By strictly synchronizing timestamps, accounting for dual-lens physical offsets, and applying a robust penalty kernel to ignore motion blur, this pipeline calculates mathematically precise visual odometry and fuses it into a high-definition 3D mesh.

## ⚙️ High-Level Pipeline
The system operates completely offline and processes raw physical data using the following pipeline:

Strict Hardware Synchronization: Reads raw RGB video frames and Infrared Depth maps, dynamically pairing them based on exact microsecond timestamps (max tolerance of 0.02s) to prevent texture smearing.

Dual-Lens Geometric Projection: Projects the RGB and Depth data into a local 3D Point Cloud using mathematically separated camera matrices (Intrinsics), accounting for the physical physical offset between the color lens and the infrared laser.

Robust Odometry (Huber Loss): Calculates frame-to-frame camera movement using Point-to-Plane ICP. Crucially, it injects a Huber Loss Kernel—a robust penalty function that mathematically acts as a guillotine, ignoring flying pixels and motion blur caused by fast camera movements.

High-Definition TSDF Mapping: Instead of blindly stacking noisy points, the tracked frames are integrated into a Scalable TSDF (Truncated Signed Distance Function) Volume. This mathematically averages sensor noise into a single sharp surface using millimeter-scale voxels (e.g., 5mm to 15mm resolution).

Mesh Extraction & Post-Processing: Extracts a solid, optimized triangle mesh from the TSDF volume, bridging microscopic gaps and outputting a rigid, presentation-ready 3D artifact.

## 📂 How to Use (Data Structure)
The system is configured out-of-the-box to accept the standard TUM RGB-D Dataset format. You must use the raw depth images, not AI-generated depth.

Your working directory must contain a dataset folder structured exactly like this:

```text
dataset_room/
│
├── rgb/                # Raw RGB frames
│   ├── 1305031102.175304.png
│   ├── 1305031102.208102.png
│   └── ...
│
└── depth/                # Raw Infrared Depth maps (Metric Scale: 5000.0)
    ├── 1305031102.196305.png
    ├── 1305031102.227401.png
    └── ...
```
(Note: The groundtruth.txt file is no longer required for the mapping phase, as the robust odometry kernel correctly tracks the trajectory, but it can be retained for evo_ape mathematical evaluation).

## 🚀 How to Run

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required mathematical and 3D processing libraries:

```bash
pip install open3d numpy scipy
```

### 2. Configure the Script
Open your main Python script and ensure the dataset paths point to your local raw data folders. To optimize for your machine's available RAM, you can adjust the TSDF resolution parameters inside the script:

voxel_length=0.015: Standard definition (fast, low memory footprint).

voxel_length=0.005: Hero-shot definition (highly photorealistic, massive memory footprint).

### 3. Execute the Pipeline
Run the script from your terminal. The script will automatically synchronize the raw hardware timestamps, execute the robust ICP loop, and begin fusing the environment.

```bash
python TUM_robust.py
```
### 4. View the Result
Once the script finishes running, Open3D will automatically launch an interactive visualization window. Use the - (minus) key on your keyboard to thin the point visualization and reveal the high-resolution geometric details!

## NOTE:
We previously used the groundtruth.txt(realtime trajectory) to map the points to 3D, this script lies in run_slam.py still and does give better results of course.