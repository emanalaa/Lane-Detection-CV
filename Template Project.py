import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import math
from scipy import interpolate

team_members_names = ['اسم الطالب باللغة العربية', ....]
team_members_seatnumbers = ['رقم الجلوس', .....]


def draw_lines_connected(img, lines, color=[255, 0, 0], thickness=8):
    # This function should draw lines to the images (default color is red and thickness is 8)

def convert_rbg_to_grayscale(img):
    # This function will do color transform from RGB to Gray
    
def convert_rgb_to_hsv(img):
    # This function will do color transform from RGB to HSV
    
def detect_edges_canny(img, low_threshold, high_threshold):
    # You should implement your Canny Edge Detector here

def remove_noise(img, kernel_size):
    # You should implement Gaussian Noise Removal Here
    
def mask_image(img, vertices):
    # Mask out the pixels outside the region defined in vertices (set the color to black)

def hough_transform(# to be determined):
    # Apply Hough transform to find the lanes

# main part

#1 read the image
#2 convert to HSV
#3 convert to Gray
#4 Threshold HSV for Yellow and White (combine the two results together)
#5 Mask the gray image using the threshold output fro step 4
#6 Apply noise remove (gaussian) to the masked gray image
#7 use canny detector and fine tune the thresholds (low and high values)
#8 mask the image using the canny detector output
#9 apply hough transform to find the lanes
#10 apply the pipeline you developed to the challenge videos

#11 You should submit your code
