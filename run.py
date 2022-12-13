import os
import cv2
import pyrealsense2 as rs
import numpy as np
from fastdlo.core import Pipeline

'''
Notation: 
before running the following commands, make sure you create the folder 'weights' 
under main program (same folder level as this script), and load the NN weights 
files (with *.pth ending) in it.
the weights checkpoints you can download from here: https://drive.google.com/file/d/1OVcro53E_8oJxRPHqGy619rBNoCD3rzT/view
'''

'''
the parameters that you may need to fine tune or utilise:
IMG_W
IMG_H
MASK_TH
---
dlo_mask_pointSet
dlo_path
'''

if __name__ == "__main__":

    #########################
    # Set up the parameters #
    #########################
    '''
    ########################################################################################
    # YOU MIGHT NEED TO TOUCH THESE PARAMETERS!
    '''
    IMG_W = 640 # image size
    IMG_H = 480
    R = np.array([[0,1],[1,0]]) # Rotation matrix for coordinate fitting
    Distance_threshold = 60 # distance detection threshold
    
    '''
    colorRange for HSV detection, 
    save in form of [((H_lower, S_lower, V_lower),(H_upper, S_upper, V_upper)),..]
    '''
    colorRange = [
        ((0, 43, 46),(10, 255, 255)), # red color range 1 HSV
        ((156, 43, 46),(180, 255, 255)) # red color range 2 HSV
        # you can add as many colors as you would like
        # final detected merged color == color 1 + color 2 + ..
    ]
    activate_interpolation = True
    '''
    #######################################################################################
    '''
    
    # weighting file name - NO NEED TO TOUCH
    ckpt_siam_name = "CP_similarity.pth"
    ckpt_seg_name = "CP_segmentation.pth"
    # get current run file path - NO NEED TO TOUCH
    script_path = os.getcwd()
    # get network weights <- need to create folder named 'weights' under main program folder, paste *.ckpt/*.pth files in it
    checkpoint_siam = os.path.join(script_path, "weights/" + ckpt_siam_name)
    checkpoint_seg = os.path.join(script_path, "weights/" + ckpt_seg_name)
    # load FASTDLO algorithm pipeline
    '''
    #############################################################################################################
    NOTATION: color Filter is added into the detection, pipeline need to transfer color range for color filtering
    '''
    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, 
    img_w=IMG_W, img_h=IMG_H, colorRange=colorRange, is_interpolation=activate_interpolation)
    # p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)
    '''
    #############################################################################################################
    '''
    
    # set up the realsense pipeline & config - NO NEED TO TOUCH
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.rgb8, 30)
    pipeline.start(config)

    ##############################
    # run the detection progress #
    ##############################
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            # run the FASTDLO, return a array of size (IMG_H, IMG_W, 3)
            img_out, dlo_mask_pointSet, dlo_path = p.run(source_img=color_image, mask_th=77)
            '''
            Interfaces:
                dlo_mask_pointSet       point set of mask region [np.array([x,y]),...]
                dlo_path                point set of dlo pose path [np.array([x,y]),...]
                img_out                 masked image/ video frame
            '''
            for i in range(len(dlo_path)):
                pos = dlo_path[i]
                pos = R.dot(pos) # coordinate btw image and path fitting
                if i != len(dlo_path)-1:
                    pos2 = dlo_path[i+1]
                    pos2 = R.dot(pos2)
                    # detect whether the points in between are too far away, 
                    # if not, then connect them with line segment
                    if np.sqrt(((pos[0]-pos2[0])**2)+((pos[1]-pos2[1])**2)) < Distance_threshold:
                        cv2.line(color_image, pt1=pos, pt2=pos2, color=(255,0,0))
                    else:
                        pass
                else:
                    pos2 = None
                # draw the path line marker
                cv2.drawMarker(color_image, tuple(pos), color=(255,0,0), markerType=3, markerSize=7, thickness=1)

            canvas = color_image.copy()
            # show the detected results with origin stream video together with weight 1.0, 0.3 for each 
            canvas = cv2.addWeighted(canvas, 1.0, img_out, 0.3, 0.0) 

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE) # set up the window to display the results
            cv2.imshow("RealSense", canvas)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27: # press the keyboard 'q' or 'esc' to terminate the thread
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()