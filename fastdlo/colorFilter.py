import cv2
import numpy as np

###############################################################################
##################### 2022.12.12 developed by Y.Meng ##########################
###############################################################################

def setMask(img, mask):
    # split the channel
    channels = cv2.split(img)
    result = []
    for i in range(len(channels)):
        result.append(cv2.bitwise_and(channels[i], mask))
        # append masking for each channel
    merged_img = cv2.merge(result)
    return merged_img

class ColorFilter():
    def __init__(self) -> None:
        # color filtering range
        # [((H_lower, S_lower, V_lower),(H_upper, S_upper, V_upper)),..]
        self.colorRange = []
    
    def __call__(self, img):
        # final generated masking
        finalMask = np.zeros_like(img)[:,:,0]

        # convert BGR frame into HSV format
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # read each HSV range limits
        for each in self.colorRange:
            lower, upper = each

            # create masking using each range
            mask = cv2.inRange(hsv, lower, upper)

            # merge the masking, target color = color 1 + color 2 ...
            finalMask = cv2.bitwise_or(finalMask, mask)
            
        # set up the final merged masked image
        merged_img = setMask(img, finalMask)
        return merged_img
