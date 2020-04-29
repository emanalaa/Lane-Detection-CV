import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import math
from scipy import interpolate
from PIL import Image
from scipy import ndimage


team_members_names = ['اسم الطالب باللغة العربية', 'ميرنا محمد يسري']
team_members_seatnumbers = ['رقم الجلوس', '2016170450']


def draw_lines_connected(img, lines, color=[255, 0, 0], thickness=8):
    # This function should draw lines to the images (default color is red and thickness is 8)
    return


def convert_rbg_to_grayscale(img):
    # This function will do color transform from RGB to Gray
    return


def convert_rgb_to_hsv(img):
    # This function will do color transform from RGB to HSV
    return


def gaussian_kernel(sigma=0.5):
    # Specify the kernel size
    if sigma > 1:
        size = 5
    else:
        size = 3

    # Calculate the formula
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = normal * np.exp(-((x**2 + y**2) / (2.0*sigma**2)))

    return g


def noise_reduction(img):
    kernel = gaussian_kernel(1.4)
    '''
    window_size = kernel.shape[0]
    offset = window_size//2
    height = img.shape[0]
    width = img.shape[1]
    # the Filter algorithm
    img2 = img.copy()
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            new_pixel = 0
            for n in range(window_size):
                for m in range(window_size):
                    a = i - offset + n
                    b = j - offset + m
                    new_pixel += img[a, b] * kernel[n, m]
            img2[i, j] = new_pixel
    '''
    img = ndimage.filters.convolve(img, kernel)
    return img


def gradient_calculation(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    G = G.astype(int)
    return G, theta


def non_max_suppression(img, theta):
    height, width = img.shape
    out = np.zeros((height, width), dtype=int)
    # Change from radian to degree
    angle = theta * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, height-1):
        for j in range(1, width-1):
            point_1 = 255
            point_2 = 255

            # Angle 0°
            # gradient magnitude is greater than the magnitudes at pixels in the (east and west) directions
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                point_1 = img[i, j-1]
                point_2 = img[i, j+1]
            # Angle 45°
            # gradient magnitude is greater than the magnitudes at pixels in the (north east and south west) directions.
            elif 22.5 <= angle[i, j] < 67.5:
                point_1 = img[i-1, j+1]
                point_2 = img[i+1, j-1]
            # Angle 90°
            # gradient magnitude is greater than the magnitudes at pixels in the (north and south) directions.
            elif 67.5 <= angle[i, j] < 112.5:
                point_1 = img[i-1, j]
                point_2 = img[i+1, j]
            # Angle 135°
            # gradient magnitude is greater than the magnitudes at pixels in the (north west and south-east) directions.
            elif 112.5 <= angle[i, j] < 157.5:
                point_1 = img[i-1, j-1]
                point_2 = img[i+1, j+1]

            # if the pixel's value is the maximum then update the out, otherwise leave it with value = 0
            if (img[i, j] >= point_1) and (img[i, j] >= point_2):
                out[i, j] = img[i, j]
    return out


def double_thresholding(img, low_threshold=2, high_threshold=25):
    # define the fixed values of the pixels
    weak = 25
    strong = 255

    # extract the indices of the pixels per condition
    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((low_threshold <= img) & (img <= high_threshold))
    irrelevant_i, irrelevant_j = np.where(img < low_threshold)

    # update the indices with the corresponding pixel value
    img[strong_i, strong_j] = strong
    img[weak_i, weak_j] = weak
    img[irrelevant_i, irrelevant_j] = 0

    return img, weak, strong


def hysteresis_edge_tracking(img, weak, strong):
    height, width = img.shape
    # Get the indices of the weak pixels
    weak_i, weak_j = np.where(img == weak)
    for (i, j) in zip(weak_i, weak_j):
        if i == 0 or j == 0 or i == height - 1 or j == width - 1:
            continue
        # Check if any of the  8-connected neighborhood pixels is a strong pixel
        if np.any(np.where(img[i-1:i+2, j-1:j+2] == strong)):
            img[i, j] = strong
        # If not, then suppress the weak pixel
        else:
            img[i, j] = 0
    return img


# Helper Function, to reduce the redundant
# plotting code
def plotting(img, fig_num, title):
    plt.figure(fig_num)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    return


def detect_edges_canny(img, low_threshold, high_threshold):
    img = img.copy()

    # 1. Noise reduction
    # using Gaussian Smoothing Kernel
    img = noise_reduction(img)
    #plotting(img, 1, 'Gaussian Filter')

    # 2. Gradient Calculations
    # using Sobel kernels
    img, theta = gradient_calculation(img)
    #plotting(img, 2, 'Sobel Filter (G)')

    # 3. Non-Maximum Suppression
    img = non_max_suppression(img, theta)
    #plotting(img, 3, 'Non-Maximum Suppression')

    # 4. Double Thresholding
    img, weak, strong = double_thresholding(img, low_threshold, high_threshold)
    #plotting(img, 4, 'Double Thresholding')

    # 5. Edge Tracking by Hysteresis
    img = hysteresis_edge_tracking(img, weak, strong)
    #plotting(img, 5, 'Edge Tracking by Hysteresis')

    #plt.show()
    return img


def remove_noise(img, kernel_size):
    # You should implement Gaussian Noise Removal Here
    return


def mask_image(img, vertices):
    # Mask out the pixels outside the region defined in vertices (set the color to black)
    return


def hough_transform():
    # Parameters to be determined
    # Apply Hough transform to find the lanes
    return


def read_image(img_path):
    img = Image.open(img_path)
    img = img.convert("L")
    img = np.asarray(img, dtype=int)
    return img


def read_video(vid_path):
    # TODO: implement this function
    return  # video_array


# main part
def main():
    # TODO: the below lines are for initial testing, remove them later
    input_image = read_image('TestCanny.jpg')
    img = detect_edges_canny(input_image, low_threshold=2, high_threshold=50)
    return

# 1 read the image
# 2 convert to HSV
# 3 convert to Gray
# 4 Threshold HSV for Yellow and White (combine the two results together)
# 5 Mask the gray image using the threshold output from step 4
# 6 Apply noise remove (gaussian) to the masked gray image
# 7 use canny detector and fine tune the thresholds (low and high values)
# 8 mask the image using the canny detector output
# 9 apply hough transform to find the lanes
# 10 apply the pipeline you developed to the challenge videos

# 11 You should submit your code


main()
