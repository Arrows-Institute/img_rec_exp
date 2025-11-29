#############################
#### Imports
#############################
from io import BytesIO

import cv2
import IPython.display
import numpy as np
import PIL.Image


#############################
#### functions
#############################
def resize_img(img: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    h_o, w_o = img.shape[:2]
    aspect = w_o / h_o
    if width / height >= aspect:
        h_n = height
        w_n = round(h_n * aspect)
    else:
        w_n = width
        h_n = round(w_n / aspect)

    return cv2.resize(img, dsize=(w_n, h_n))


# identify cone
def identify_object(img: np.ndarray, hsv_lower: np.ndarray, hsv_upper: np.ndarray, exist_size: int) -> None:
    img_cp = np.copy(img)

    def find_contours(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_lower[0] /= 2
        hsv_upper[0] /= 2
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask

    def check_contours_exist(contours: np.ndarray) -> tuple[bool, np.ndarray | None]:
        greatest_contour = None

        if not contours:
            return (False, greatest_contour)
        greatest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(greatest_contour) < exist_size:
            return (False, greatest_contour)

        return (True, greatest_contour)

    contours, _ = find_contours(img_cp)
    contours_exist, greatest_contour = check_contours_exist(contours)
    if contours_exist:
        cv2.drawContours(img_cp, [greatest_contour], 0, (0, 255, 0), 3)

    img = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
    f = BytesIO()
    PIL.Image.fromarray(img).save(f, "jpeg")
    img_jpeg = IPython.display.Image(data=f.getvalue())
    IPython.display.display_jpeg(img_jpeg)
