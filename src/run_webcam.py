import argparse
import time
import numpy as np
from My_func import create_dict, manipulate
import cv2
import os
from Feature_Extraction import Get_Coords, Calculate_D, Calculate_L, Calculate_PCM, Calculate_TCM, Calculate_R, Add_Features_To_dataframe


from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh


fps_time = 0

#####################################################################
###################### INPUT VIDEO DIRECTORY ########################
#####################################################################

## # NOTES:
# Please put your video in a folder inside the INPUT_DIR path
#For example D:\\videos\\daria where daria is a folder containing the video/s
INPUT_DIR='D:\\HANDSA\\Freelance\\new'  
folders=os.listdir(INPUT_DIR)
CURRENT_PATH=os.getcwd()

######################### FLAGS ##################################
SAVE_FRAMES=1 # Change to 1 if you want to save the frames
SHOW_FRAMES_WHILE_COMPUTING = 0 # Flag to show images while computing the features
#####################################################################
#####################################################################
#####################################################################
index=0
for folder_name in folders:
    new_path=INPUT_DIR+'\\'+folder_name 
    files_list=os.listdir(new_path)
    
    newpath=CURRENT_PATH+'\Output'
    if not os.path.exists(newpath):
        os.makedirs(newpath)  
        
    for file in files_list:
                
  
        FILE_NAME=file      
        FOLDER_NAME=file.split('_')[0]        
        

        Pose_Persons={}
        
        if __name__ == '__main__':
            parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
            parser.add_argument('--camera', type=int, default=0)
            parser.add_argument('--zoom', type=float, default=1.0)
            parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
            parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
            parser.add_argument('--show-process', type=bool, default=False,
                                help='for debug purpose, if enabled, speed for inference is dropped.')
            args = parser.parse_args()
        
        
            w, h = model_wh(args.resolution)
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        
            cam = cv2.VideoCapture(INPUT_DIR+'\\'+folder_name+'\\'+file)             # Captures the video 
            ret_val, image = cam.read()                   # Takes first frame
        
            image_shape_x=image.shape[0]
            image_shape_y=image.shape[1]
            
            images=[]
            begin_counting=time.time()
            Frame_n=0
        
            while True:
                ret_val, image = cam.read()
                
                if image is None:break
                
                print('Frame number = ',Frame_n)
                print("image shape = {} , image type = {}".format(image.shape, type(image))) 
        
                if args.zoom < 1.0:
                    canvas = np.zeros_like(image)
                    img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                    dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
                    dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
                    canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
                    image = canvas
                elif args.zoom > 1.0:
                    img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                    dx = (img_scaled.shape[1] - image.shape[1]) // 2
                    dy = (img_scaled.shape[0] - image.shape[0]) // 2
                    image = img_scaled[dy:image.shape[0], dx:image.shape[1]]
        
        
                humans, p1_id, p2_id, x1,x2,y1,y2 = e.inference(image, image_shape_x, image_shape_y)
                # Computes the estimated points for each human in the picture one by one
                if not p1_id[0] == -500 :
                    Pose_Persons[Frame_n]=create_dict(p1_id, p2_id, x1, x2, y1, y2) 
                    Frame_n+=1
                
        
                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)   # Drawing of the estimated human
        
        
                cv2.putText(image,                    # Appends the FPS on top left
                            "FPS: %f" % (1.0 / (time.time() - fps_time)),
                            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                
                if SHOW_FRAMES_WHILE_COMPUTING == 1:           
                    cv2.imshow('tf-pose-estimation result', image)   # To see the frames while computing
                    
                fps_time = time.time()
                if cv2.waitKey(1) == 27:
                    break
        
                images.append(image)
            images=np.reshape(np.array(images), (len(images), images[0].shape[0], images[0].shape[1], 3))
        
            
            if SAVE_FRAMES==1:
                newpath=CURRENT_PATH+'\Output'+'\\'+FOLDER_NAME+'\\'+FILE_NAME[:-4]+'_images'
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                          
                for i in range (len(images)):
                    cv2.imwrite('Output\\'+FOLDER_NAME+'\\'+FILE_NAME[:-4]+'_images'+'\\'+FILE_NAME[:-4]+'_img'+str(i)+'.png',images[i])
                print('')
                print('Successfully saved the frames!')
            
            if SHOW_FRAMES_WHILE_COMPUTING == 1:    
                cv2.destroyAllWindows()
        
        
        print('')
        print('Time for each operation in ms for the file '+FILE_NAME.replace('.csv','')+' :')
        print('---------------------------------------------------------------------------------')
        
        Co_ordinates=manipulate(Pose_Persons)     # Computes the Co-ordinates from the given data by O-Nect
        print('Time taken for Co-ordinate calculations for {} Frames is {:.3f} ms'.format(len(Co_ordinates),(time.time()-begin_counting)*1000))
        temp=Co_ordinates.copy()
        # Extracting X, Y Coordinates
        start=time.time()
        X_Coords, Y_Coords= Get_Coords(Co_ordinates)
        print('Time taken for Co-ordinate extractions for {} Frames is {:.3f} ms'.format(len(Co_ordinates),(time.time()-start)*1000))
        
        start=time.time()
        PCM_Frames= Calculate_PCM(X_Coords, Y_Coords)
        print('Time taken for PCM for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
        
        start=time.time()
        TCM_x, TCM_y= Calculate_TCM(PCM_Frames)
        print('Time taken for TCM for {} Frames is {:.3f} ms'.format(len(TCM_x),(time.time()-start)*1000))
        
        start=time.time()
        L= Calculate_L(TCM_x, TCM_y, PCM_Frames)
        print('Time taken for L features for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
        
        start=time.time()
        D1, D2, D3 = Calculate_D(PCM_Frames, TCM_x, TCM_y, 'Degrees')
        print('Time taken for D1, D2, D3 features for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
        
        start=time.time()
        R=Calculate_R(PCM_Frames)
        print('Time taken for R feature for {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-start)*1000))
        
        start=time.time()
        out=Add_Features_To_dataframe(FILE_NAME[:-4]+'_'+str(Frame_n-1),temp,Co_ordinates, PCM_Frames, TCM_x, TCM_y, L, R, D1, D2, D3, FOLDER_NAME)
        print('Time taken for adding features to dataframe for {} Frames is {:.3f} ms'.format(len(Co_ordinates),(time.time()-start)*1000))
        
        if index==0:
            F_out=out.copy()
            index=1
        else:
            F_out=F_out.append(out)
        
        print('Time taken the for whole file of {} Frames is {:.3f} ms'.format(len(PCM_Frames),(time.time()-begin_counting)*1000))
        
        
newpath=CURRENT_PATH+'\Output'+'\\'+FOLDER_NAME
if not os.path.exists(newpath):
    os.makedirs(newpath)
OUTPUT_PATH = 'Output\\'+'Final_Output'+'.csv'
F_out.to_csv(OUTPUT_PATH, index=False)
