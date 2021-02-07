import numpy as np
import cv2 as cv


def local_constrast_enhancement(img):
    h, w = img.shape
    img = img.astype(np.float32)

    meanV = cv.blur(img, (15, 15))
    normalized = img - meanV
    var = abs(normalized)

    var = cv.blur(var, (15, 15))

    normalized = normalized / (var + 10) * 0.75
    normalized = np.clip(normalized, -1, 1)
    normalized = (normalized + 1) * 127.5
    return normalized

def custom_binarization(img):
    blur = cv.GaussianBlur(img,(3,3),0)
    blur = cv.GaussianBlur(blur,(3,3),0)
    img_proc = local_constrast_enhancement(blur)
    
    hist = cv.calcHist([img_proc],[0],None,[256],[0,256])
    
    peak_idx = np.argmax(hist)
    indxes = np.where(hist - 20 > 0)[0]
    shift = peak_idx - indxes[0]
    tresh = peak_idx + shift
    bin_img = (img_proc > tresh) * 255
    return bin_img.astype(np.uint8)
