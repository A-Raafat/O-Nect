# Server.py to be run in unity or blender
#set  blender on game engine and create keypoints using spheres
#access these in real time to get body pose based movement
#import bpy
import socket
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from My_func import create_dict, manipulate
import cv2
import os
from Feature_Extraction import Get_Coords, Get_Mass, Calculate_D, Calculate_L, Calculate_PCM, Calculate_TCM, Calculate_R, Add_Features_To_dataframe
import time

host = "127.0.0.1"
port = 1234
buffer_size = 1024

my_socket = socket.socket()
my_socket.bind( (host, port) )

my_socket.listen(1);		# listen indefinately
conn, addr = my_socket.accept()
print("Connected to client at :"+str(addr))

# since we read buffer_size bytes every time, we need the client to send just 1 json string each time, else we may read many buffered strings and json.loads() gives adn error, hence synchronization is needed.
start='1'
stop='0'

Pose_Persons={}
j=0

#####################################################################
###################### INPUT VIDEO DIRECTORY ########################
INPUT_DIR='D:\\HANDSA\\Freelance\\videos\\P14\\person05_jogging_d2_uncomp.avi'  # Must be in this format
MAX_FRAME_NUMBER=2000                  # Maximum desired number of frames
#####################################################################
#####################################################################

FILE_NAME=INPUT_DIR.split('\\')[len(INPUT_DIR.split('\\'))-1][:-4]+'.csv'  # Splits the directory and takes the name of the file
FOLDER_NAME=FILE_NAME.split('_')[0] # splits the name and takes the first part of the name

CURRENT_PATH=os.getcwd()
############ Create folder ################
newpath=os.getcwd()+'\Output'+'\\'+FOLDER_NAME
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
OUTPUT_PATH=newpath+'\\'+ FILE_NAME

# get iamge shape
cam = cv2.VideoCapture(INPUT_DIR)      # Loads the vid
ret_val, image = cam.read()            # Reads first frame
image_shape_x=image.shape[0]           # Saves the shape   
image_shape_y=image.shape[1]

begin_counting=time.time()             # For time computing
data = conn.recv(buffer_size).decode() # Recieves and decodes the message


for i in range(MAX_FRAME_NUMBER):
  
    
    if not data:
        break
    if data =='start':                 # 'start' is a flag sent by run_webcam indicating the start of frame 
        p1_id=[]
        p2_id=[]
    
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        data=''
        
        while not data=='start':       
            data = conn.recv(buffer_size).decode()  
            #print(data)
            if (not data[0:5]=='start') & (not data==''):  # Empty string means the end of the frame, so it loops till the end of frame while there is start of frame
                data = json.loads(data)
            
                conn.send("ACK".encode())		# For synchronization of data sent between client server
                #print("{0}: ({1},{2})\n".format(data['p1_id'], data['x1_coord'], data['y1_coord']))
                #print("{0}: ({1},{2})\n".format(data['p2_id'], data['x2_coord'], data['y2_coord']))
                
                p1_id.append(data['p1_id'])
                p2_id.append(data['p2_id'])      # Just appending the values
                
                x1.append(data['x1_coord']*image_shape_x)
                x2.append(data['x2_coord']*image_shape_x)
                y1.append(data['y1_coord']*image_shape_y)
                y2.append(data['y2_coord']*image_shape_y)
            else:break
    print("Frame number {} success!".format(i+1))
    p1_id=np.array(p1_id)
    p2_id=np.array(p2_id)
    
    x1=np.array(x1)
    x2=np.array(x2)
    y1=np.array(y1)
    y2=np.array(y2)
    
    
    Pose_Persons[j]=create_dict(p1_id, p2_id, x1, x2, y1, y2)      # Creating a dictionary numbered by frame number and contains the values of p1_id, p2_id, x1,x2, y1,y2
    j+=1
                

conn.close()      # Closes connection

