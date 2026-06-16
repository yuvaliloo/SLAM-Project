# Depth-SLAM

## 💡 Project Idea

Traditional SLAM (Simultaneous Localization and Mapping) algorithms struggle with two compounded problems: guessing the depth of a scene from a 2D image, and calculating the camera's movement. When either guess is slightly wrong, the errors multiply, causing geometric drift and "ghosting" in the final 3D map.

This project implements an Offline SLAM Architecture that separates these two tasks to eliminate drift. It fuses mathematically perfect camera trajectory data (Ground Truth from IMU/Motion Capture) with state-of-the-art, zero-shot AI depth estimation (Depth Anything V2). By letting the hardware handle the localization and the AI handle the perception, we can construct dense, highly accurate 3D point clouds of indoor environments without algorithmic tracking failures.

## ⚙️ High-Level Pipeline

The system operates completely offline and processes data using the following pipeline:

Data Synchronization: Reads raw RGB video frames and dynamically pairs them with the closest matching timestamp in the hardware trajectory database.

AI Perception (In-Memory): Passes the 2D RGB image into Depth Anything V2 (running on the GPU) to hallucinate a metric 2.5D depth map dynamically.

Geometric Projection: Projects the RGB and AI Depth data into a local 3D Point Cloud using the camera's specific physical lens matrix (Intrinsics).

ICP Micro-Alignment: Uses the absolute Ground Truth trajectory as an initial placement guess, then applies Iterative Closest Point (ICP) math to "snap" the new frame into the master map, correcting any temporal wobbling from the AI's depth estimations.

Global Mapping: Merges the aligned frame into a master Point Cloud and exports a single, clean .ply file.

## 📂 How to Use (Data Structure)

The system requires an image sequence and a trajectory file. It is configured out-of-the-box to accept the standard TUM RGB-D Dataset format.

Your working directory must contain a dataset folder structured exactly like this:
```text
dataset_room/
│
├── rgb/                      # Frames folder
│   ├── 1305031102.175.png    # (Named by timestamp)
│   ├── 1305031102.208.png
│   └── ...
│
└── groundtruth.txt           # Trajectory file
``` 
provided by external tracking/IMU


Note on groundtruth.txt: The text file must contain space-separated values in the following format:
timestamp tx ty tz qx qy qz qw (Position X, Y, Z and Quaternion Rotations).

(Note: The system dynamically calculates depth, so no depth/ folder is required!)

## 🚀 How to Run

### 1. Install Dependencies

Ensure you have Python 3.8+ installed. Install the required mathematical, computer vision, and AI libraries:
```bash
pip install -r requirements.txt
```

### 2. Configure the Script

Open Open3D_noDepth.py and ensure the dataset_dir variable at the bottom points to your local data folder:

dataset_dir = "dataset_room"  # Change to your folder name


### 3. Execute the Pipeline

Run the script from your terminal. The script will load the AI model onto your GPU, parse the trajectory, and begin fusing frames.
```bash
python Open3D_noDepth.py
```

### 4. View the Result

Once the script finishes running, it will generate a file named tum_desk_ai_icp.ply in your root directory.
Drag and drop this file into CloudCompare or MeshLab to navigate your newly reconstructed 3D environment!