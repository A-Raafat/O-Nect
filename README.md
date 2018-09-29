# CM calculations and feature extraction for ensemble learning using O-Nect Library
O-Nect is an Open Source Interface for Unity Game developers which uses deep learning (human pose estimation) along with socket connectivity which provides kinect similar performance with just a regular RGB camera.

Yes,No sensors required.


<p align="center">
    <img src="https://github.com/O-Nect/O-Nect/blob/master/models/O-nect.gif", width="480">
</p>


Inspired by the work of OpenPose Developers(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose
We believe that this holds alot of potential and the users can manipulate the code as per their convenience.

You need dependencies below.
- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk
- socket,json,threading

use : $pip install -r requirements.txt
and Blender

Once you've got everything ready
Run this command for RealTime Webcam 


$ python run_webcam.py --camera=0
You can now use socket based connectivity in Unity to access the keyPoints in real-time to use it as per your game design.

We have tested the performance to be around 15fps on a gtx 1050Ti 4gb GPU on blender game engine using socket in python.
The user can use unity or any other game engine with socket support and implement the desired module.

We send cordinates using json + sockets .The basic layout of the json file is given below

json data format
{
data['p1_id'], data['x1_coord'], data['y1_coord']
data['p2_id'], data['x2_coord'], data['y2_coord']
}

We welcome contributers to test the product.
We've now added a blend file to our repo.
Run the blender script and run_webcam.py from src 
(Unity Support To be added soon)

We currently only support blender and are looking for contributers to port this to unity3D.

# List of features:

|Name|#Features|Description|
|:---:|:----:|-----------|
|X,Y|36|Co-ordinate points for Nose, Ears, Head, Eyes, Shoulders, Elbows, Wrists, Hips, Knees and Ankles|
|PCM|28|Co-ordinates for Head & Neck, Trunk, Upper Arms, Forearms, Hands, Thighs, Shank, Foot|
|TCM|2|The mid point of the body|
|L|14|Distances from TCM to all PCMs|
|R|14|The change of PCM co-ordinate points from a frame to another|
|D1|1|Angle between left and right foot from the vertex TCM|
|D2|1|Angle between TCM and left arm from the vertex left forearm|
|D3|1|Distance from TCM to the straight line which is formed by PCMs of feet|
|SF(not included)|2|Mean and standard deviation of the skeletal length|

# The Feature extraction steps:
Note: This feature extraction works best for 1 person in a video.

Step 1: Using O-Nect library to extract the points as in the data mentioned above

Step 2: Manipulating these points and collecting all the points for all frames of the videos

Step 3: Using these points to extract the PCM, TCM, L, R, D Features and placing all the coordinates in a dataframe

## Steps to run the code:

1- Open run_webcam in src folder and write the input directory in INPUT_DIR, it should be a path for the folder that contains all the video folders containing the files.

2- Set the flag SAVE_FRAMES to 1, if you want to save the images of the frames.

3- Set the flag SHOW_FRAMES_WHILE_COMPUTING to 1, if you want to see the frames while computing.

## Output
The csv files will be created in a folder named Output in the same directory of the codes (src)


Notes:
-----
1- The input directory must be in this format :  Drive:\\Folder1\\videos

