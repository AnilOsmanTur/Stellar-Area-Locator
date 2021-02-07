# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:37:23 2021

@author: anilosmantur
"""


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import preprocessing as pre

# takes image to search and the target image and returns 2d 4 location point in target image
def locate_in_stars(img_search, target_img):
    w, h = img_search.shape[::-1]
    
    # task specific image binarization
    temp_bin_img = pre.custom_binarization(img_search)
    org_bin_img = pre.custom_binarization(target_img)

    res = cv.matchTemplate(org_bin_img,temp_bin_img, cv.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    
    top_left = max_loc
    top_right = (top_left[0], top_left[1] + h)
    bot_left = (top_left[0] + w, top_left[1])
    bot_right = (bot_left[0], bot_left[1] + h)
    
    return (top_left, top_right,
            bot_left, bot_right)


def example_run(path_search_img='Small_area_rotated.png', path_target_img='StarMap.png'):
    img = cv.imread(path_target_img, cv.IMREAD_GRAYSCALE)
    temp_img = cv.imread(path_search_img, cv.IMREAD_GRAYSCALE)

    out = locate_in_stars(temp_img, img)
    (p1, p2, p3, p4) = out
    print(out)
    
    cv.rectangle(img, p1, p4, 255, 2)
    plt.figure()
    plt.imshow(img,cmap = 'gray')
    plt.show()
    cv.imwrite('result_img_'+path_search_img, img)
    
if __name__ == '__main__':
    print('hi')
    example_run(path_search_img='Small_area_rotated.png', path_target_img='StarMap.png')
    example_run(path_search_img='Small_area.png', path_target_img='StarMap.png')