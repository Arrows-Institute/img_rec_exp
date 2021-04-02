#############################
#### Imports
#############################
import cv2
import IPython.display
from ipywidgets import widgets
import numpy as np
from io import BytesIO
import PIL.Image

#############################
#### functions
############################# 
def resize_img(img, width=640, height=480):
    h_o, w_o = img.shape[:2]
    aspect = w_o/h_o
    if width/height >= aspect:
        h_n = height
        w_n = round(h_n * aspect)
    else:
        w_n = width
        h_n = round(w_n / aspect)
    
    img_re = cv2.resize(img, dsize=(w_n, h_n))
    
    return img_re

# identify cone
def identify_object(img, HSV_LOWER, HSV_UPPER, EXIST_SIZE):
    img_cp = np.copy(img)
    
    def find_contours(img):
        hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        contours,inv=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask
    
    def check_contours_exist(contours):
        greatest_contour = None
    
        if not contours:
            return (False, greatest_contour)
        greatest_contour = max(contours, key = cv2.contourArea)
    
        if cv2.contourArea(greatest_contour)<EXIST_SIZE:
            return (False,greatest_contour)
    
        return (True, greatest_contour)
    
    contours, mask = find_contours(img_cp)
    contours_exist, greatest_contour = check_contours_exist(contours)
    if contours_exist:
        cv2.drawContours(img_cp, [greatest_contour], 0, (0, 255, 0), 3)
    
    img = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
    f = BytesIO()
    PIL.Image.fromarray(img).save(f, 'jpeg')
    img_jpeg = IPython.display.Image(data=f.getvalue())
    IPython.display.display_jpeg(img_jpeg)