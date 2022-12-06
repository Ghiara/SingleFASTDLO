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
    # image size - YOU MIGHT NEED TO TOUCH THIS PARAMETERS！
    IMG_W = 640
    IMG_H = 480
    MASK_TH = 77
    # weighting file name - NO NEED TO TOUCH
    ckpt_siam_name = "CP_similarity.pth"
    ckpt_seg_name = "CP_segmentation.pth"

    # get current run file path - NO NEED TO TOUCH
    script_path = os.getcwd()
    # get network weights <- need to create folder named 'weights' under main program folder, paste *.ckpt/*.pth files in it
    checkpoint_siam = os.path.join(script_path, "weights/" + ckpt_siam_name)
    checkpoint_seg = os.path.join(script_path, "weights/" + ckpt_seg_name)
    # load FASTDLO algorithm pipeline - NO NEED TO TOUCH
    p = Pipeline(checkpoint_siam=checkpoint_siam, checkpoint_seg=checkpoint_seg, img_w=IMG_W, img_h=IMG_H)

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
            img_out, dlo_mask_pointSet, dlo_path = p.run(source_img=color_image, mask_th=MASK_TH)
            '''
            Input:
                MASK_TH:            the dlo detection threshold, only region value greater than this threshold can be identified as dlo region
            Output:
                img_out:            masked video frame with green color
                dlo_mask_pointSet:  point set that build mask region in the frame [array[x,y],..]
                dlo_path:           the point set to indicate the pose of DLO [array[x,y],..]
            '''
            
            canvas = color_image.copy()
            # show the detected results with origin stream video together with weight 0.5 for each 
            canvas = cv2.addWeighted(canvas, 0.5, img_out, 0.5, 0.0) 

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE) # set up the window to display the results
            cv2.imshow("RealSense", canvas)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27: # press the keyboard 'q' or 'esc' to terminate the thread
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()